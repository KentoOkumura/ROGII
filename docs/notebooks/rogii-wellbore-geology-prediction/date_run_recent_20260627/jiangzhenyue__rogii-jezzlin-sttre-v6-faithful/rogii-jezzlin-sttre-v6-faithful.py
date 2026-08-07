# --- cell 0 ---
#!/usr/bin/env python3
# =============================================================================
# STTRE-Rogii v6 Full-Well CV Submission Script
# CPU-only inference for code-competition submission compatibility
# Loads the v6 full-well CV model weights from Kaggle Dataset.
# =============================================================================

import os
import math
import copy
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# =============================================================================
# CUDA / PyTorch compatibility bootstrap (same as training kernel)
# =============================================================================

def _ensure_cuda_compatible():
    print(f"PyTorch version: {torch.__version__}")
    print(f"PyTorch CUDA: {torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU capability: {torch.cuda.get_device_capability(0)}")
    else:
        print("WARNING: CUDA not available, will run on CPU")
        return

    try:
        _ = torch.arange(8, device="cuda")
        print("CUDA basic operation OK")
        return
    except Exception as e:
        print(f"CUDA basic operation failed: {e}")

    if os.environ.get("ROGII_PT_RETRIED") == "1":
        raise RuntimeError("PyTorch reinstall was already attempted but CUDA still fails. Aborting.")

    print("Attempting to install PyTorch 2.5.1+cu121 (broad GPU compatibility)...")
    import subprocess
    import sys
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "--upgrade",
            "torch==2.5.1+cu121", "torchvision", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cu121",
        ])
    except subprocess.CalledProcessError as pe:
        raise RuntimeError(f"PyTorch reinstall failed: {pe}")

    print("PyTorch reinstalled. Restarting script...")
    sys.stdout.flush()
    os.environ["ROGII_PT_RETRIED"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

_ensure_cuda_compatible()

from scipy.spatial import cKDTree
from numba import njit

warnings.filterwarnings("ignore")

# =============================================================================
# Config
# =============================================================================

class CFG:
    MODEL_DIR = Path("/kaggle/input/datasets/jezzlin/rogii-sttre-v6-fullwellcv/sttre_rogii_v6_fullwellcv")
    DATA_DIR = Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction")
    OUTPUT_PATH = "/kaggle/working/submission.csv"
    DEVICE = torch.device("cpu")

    ESTIMATORS = [
        "pf_ancc", "pf_z", "beam_cons", "beam_sm5",
        "sc_ens", "dtw_ens", "tvtF_ANCC", "tvt_dense",
    ]
    N_ESTIMATORS = len(ESTIMATORS)
    MAX_SEQ_LEN = 256

    FORMATIONS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]
    BEAMS = [
        (10, 20.0, 144.0, 2, "cons"),
        (10,  8.0,  64.0, 2, "loose"),
        ( 8, 35.0, 220.0, 1, "vcons"),
        (10, 14.0,  90.0, 5, "sm5"),
        (20,  4.0,  36.0, 3, "vloose"),
        (12, 12.0, 100.0, 3, "mid"),
        (15, 25.0, 180.0, 2, "stiff"),
    ]
    PF_N = 600
    ANCC_N = 600
    PF_MOM = 0.993
    PF_VN = 0.005
    PF_PN = 0.01
    PF_GR_SIG_MIN = 10.0
    PF_GR_SIG_MAX = 60.0
    PF_GR_SIG_DEF = 30.0
    PF_INIT_V_STD = 0.02
    PF_INIT_SPR = 0.5
    PF_RESAMP = 0.5
    PF_ROUGH_P = 0.2
    PF_ROUGH_V = 0.003
    PF_GR_WIN = 5
    PF_GR_WT = 0.3
    ANCC_ALPHA = 0.998
    ANCC_RN = 0.002
    ANCC_PN = 0.005
    ANCC_IR = 0.01
    ANCC_IS = 0.3
    ANCC_RP = 0.1
    ANCC_RR = 0.001
    DTW_RADII = (20, 50, 100, 200)
    PLANE_K = 10
    DENSE_SPW = 60
    DENSE_K = 20


# =============================================================================
# Numba JIT Functions
# =============================================================================

@njit(cache=True)
def _interp1(grid, v, vmin, step):
    i = int((v - vmin) / step)
    if i < 0:
        return grid[0]
    n = len(grid) - 1
    if i >= n:
        return grid[n]
    t = (v - vmin) / step - i
    return grid[i] * (1.0 - t) + grid[i + 1] * t


@njit(cache=True)
def _resamp(pos, aux, w, N, rp, rv):
    cum = np.zeros(N + 1)
    for j in range(N):
        cum[j + 1] = cum[j] + w[j]
    u0 = np.random.uniform(0.0, 1.0 / N)
    np2 = np.empty(N)
    na = np.empty(N)
    ci = 0
    for j in range(N):
        u = u0 + j / N
        while ci < N - 1 and cum[ci + 1] < u:
            ci += 1
        np2[j] = pos[ci] + rp * np.random.randn()
        na[j] = aux[ci] + rv * np.random.randn()
    return np2, na


@njit(cache=True)
def _beam_jit(sgr, tw_gr, si, BS, mc, es):
    n = len(sgr)
    nt = len(tw_gr)
    MAX = BS * 6
    bidx = np.zeros(BS, np.int64)
    bidx[0] = si
    bcost = np.full(BS, 1e30)
    bcost[0] = 0.0
    bn = np.int64(1)
    hI = np.zeros((n, BS), np.int64)
    hP = np.zeros((n, BS), np.int64)
    cI = np.zeros(MAX, np.int64)
    cC = np.full(MAX, 1e30)
    cP = np.zeros(MAX, np.int64)
    for step in range(n):
        gv = sgr[step]
        nc = np.int64(0)
        for bi in range(bn):
            idx = bidx[bi]
            cost = bcost[bi]
            for d in range(-2, 3):
                ni = idx + d
                if ni < 0 or ni >= nt:
                    continue
                tot = cost + (gv - tw_gr[ni]) ** 2 / es + mc * (d if d >= 0 else -d)
                fnd = np.int64(-1)
                for ci_ in range(nc):
                    if cI[ci_] == ni:
                        fnd = ci_
                        break
                if fnd >= 0:
                    if tot < cC[fnd]:
                        cC[fnd] = tot
                        cP[fnd] = bi
                else:
                    if nc < MAX:
                        cI[nc] = ni
                        cC[nc] = tot
                        cP[nc] = bi
                        nc += 1
        kept = min(BS, nc)
        for i in range(kept):
            mi = i
            for j in range(i + 1, nc):
                if cC[j] < cC[mi]:
                    mi = j
            if mi != i:
                cI[i], cI[mi] = cI[mi], cI[i]
                cC[i], cC[mi] = cC[mi], cC[i]
                cP[i], cP[mi] = cP[mi], cP[i]
        hI[step, :kept] = cI[:kept]
        hP[step, :kept] = cP[:kept]
        bidx[:kept] = cI[:kept]
        bcost[:kept] = cC[:kept]
        bn = kept
    best = np.int64(0)
    for b in range(1, bn):
        if bcost[b] < bcost[best]:
            best = b
    path = np.zeros(n, np.int64)
    b = best
    for s in range(n - 1, -1, -1):
        path[s] = hI[s, b]
        b = hP[s, b]
    return path


@njit(cache=True)
def _dtw_sakoe_chiba(query, ref, radius):
    N = len(query)
    M = len(ref)
    INF = 1e18
    D = np.full((N, M), INF)
    slope = (M - 1.0) / max(N - 1.0, 1.0)
    for i in range(N):
        j_center = int(round(i * slope))
        j_lo = max(0, j_center - radius)
        j_hi = min(M - 1, j_center + radius)
        for j in range(j_lo, j_hi + 1):
            cost = (query[i] - ref[j]) ** 2
            if i == 0 and j == 0:
                D[i, j] = cost
            elif i == 0:
                prev = D[i, j - 1]
                D[i, j] = cost + (prev if prev < INF else INF)
            elif j == 0:
                prev = D[i - 1, j]
                D[i, j] = cost + (prev if prev < INF else INF)
            else:
                a = D[i - 1, j - 1]
                b = D[i - 1, j]
                c = D[i, j - 1]
                mn = a if a < b else b
                mn = mn if mn < c else c
                D[i, j] = cost + (mn if mn < INF else INF)
    i = N - 1
    j = M - 1
    pi = np.zeros(N + M, np.int64)
    pj = np.zeros(N + M, np.int64)
    k = 0
    while i > 0 or j > 0:
        pi[k] = i
        pj[k] = j
        k += 1
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            a = D[i - 1, j - 1]
            b = D[i - 1, j]
            c = D[i, j - 1]
            if a <= b and a <= c:
                i -= 1
                j -= 1
            elif b <= c:
                i -= 1
            else:
                j -= 1
    pi[k] = 0
    pj[k] = 0
    k += 1
    return D, pi[:k], pj[:k]


@njit(cache=True)
def _dtw_path_to_tvt(pi, pj, tw_tvt, N):
    j_for_i = np.zeros(N, np.int64)
    for k in range(len(pi)):
        j_for_i[pi[k]] = pj[k]
    result = np.empty(N, np.float32)
    for i in range(N):
        result[i] = tw_tvt[j_for_i[i]]
    return result


@njit(cache=True)
def _pf_ancc(md_v, z_v, gr_v, gg, vmin, step, gs, ls, ir, N,
             ALPHA, RN, PN, IS, RP, RR, RESAMP):
    pos = np.empty(N)
    rate = np.empty(N)
    w = np.ones(N) / N
    for j in range(N):
        pos[j] = ls + IS * np.random.randn()
        rate[j] = ir + 0.01 * np.random.randn()
    pts = np.empty(len(md_v))
    std_ = np.empty(len(md_v))
    pm = md_v[0] - 1.0
    for i in range(len(md_v)):
        dm = md_v[i] - pm
        dm = max(dm, 1.0)
        for j in range(N):
            rate[j] = ALPHA * rate[j] + RN * np.random.randn()
            pos[j] += rate[j] * dm + PN * np.random.randn()
            tvt_j = pos[j] - z_v[i]
            tvt_j = max(tvt_j, vmin - 50.0)
            tvt_j = min(tvt_j, vmin + len(gg) * step + 50.0)
            pos[j] = tvt_j + z_v[i]
        if not np.isnan(gr_v[i]):
            ws = 0.0
            for j in range(N):
                eg = _interp1(gg, pos[j] - z_v[i], vmin, step)
                d = (gr_v[i] - eg) / gs
                lk = max(np.exp(-0.5 * d * d) if d * d < 600.0 else 0.0, 1e-300)
                w[j] *= lk
                ws += w[j]
            if ws > 0.0:
                for j in range(N):
                    w[j] /= ws
            else:
                for j in range(N):
                    w[j] = 1.0 / N
        ne = 0.0
        for j in range(N):
            ne += w[j] * w[j]
        if 1.0 / ne < RESAMP * N:
            pos, rate = _resamp(pos, rate, w, N, RP, RR)
            for j in range(N):
                w[j] = 1.0 / N
        tv = 0.0
        for j in range(N):
            tv += w[j] * (pos[j] - z_v[i])
        pts[i] = tv
        va = 0.0
        for j in range(N):
            va += w[j] * (pos[j] - z_v[i] - tv) ** 2
        std_[i] = va ** 0.5
        pm = md_v[i]
    return pts, std_


@njit(cache=True)
def _pf_z(md_v, z_v, gr_v, gr_sm_v, gg_p, gg_s, vmin, step,
          gs, ip, iv, beta, icpt, zsig, N,
          MOM, VN, PN, GR_WT, RP, RV, RESAMP):
    pos = np.empty(N)
    vel = np.empty(N)
    w = np.ones(N) / N
    for j in range(N):
        pos[j] = ip + 0.5 * np.random.randn()
        vel[j] = iv + 0.02 * np.random.randn()
    pts = np.empty(len(md_v))
    std_ = np.empty(len(md_v))
    pm = md_v[0] - 1.0
    pz = z_v[0] - 1.0
    for i in range(len(md_v)):
        dm = md_v[i] - pm
        dm = max(dm, 1.0)
        dzd = (z_v[i] - pz) / dm
        ve = beta * dzd + icpt
        for j in range(N):
            vel[j] = MOM * vel[j] + VN * np.random.randn()
            pos[j] += vel[j] * dm + PN * np.random.randn()
            pos[j] = max(pos[j], vmin - 50.0)
            pos[j] = min(pos[j], vmin + len(gg_p) * step + 50.0)
        if not np.isnan(gr_v[i]):
            ws = 0.0
            for j in range(N):
                ep = _interp1(gg_p, pos[j], vmin, step)
                dp = (gr_v[i] - ep) / gs
                lp = max(np.exp(-0.5 * dp * dp) if dp * dp < 600.0 else 0.0, 1e-300)
                if not np.isnan(gr_sm_v[i]):
                    es = _interp1(gg_s, pos[j], vmin, step)
                    ds = (gr_sm_v[i] - es) / (gs * 1.5)
                    ls_ = max(np.exp(-0.5 * ds * ds) if ds * ds < 600.0 else 0.0, 1e-300)
                    lk = (1.0 - GR_WT) * lp + GR_WT * ls_
                else:
                    lk = lp
                lk = max(lk, 1e-300)
                w[j] *= lk
                ws += w[j]
            if ws > 0.0:
                for j in range(N):
                    w[j] /= ws
            else:
                for j in range(N):
                    w[j] = 1.0 / N
        ws2 = 0.0
        for j in range(N):
            dv = (vel[j] - ve) / max(zsig * 2.0, 0.005)
            lz = max(np.exp(-0.5 * dv * dv) if dv * dv < 600.0 else 0.0, 1e-300)
            w[j] *= lz
            ws2 += w[j]
        if ws2 > 0.0:
            for j in range(N):
                w[j] /= ws2
        else:
            for j in range(N):
                w[j] = 1.0 / N
        ne = 0.0
        for j in range(N):
            ne += w[j] * w[j]
        if 1.0 / ne < RESAMP * N:
            pos, vel = _resamp(pos, vel, w, N, RP, RV)
            for j in range(N):
                w[j] = 1.0 / N
        wm = 0.0
        for j in range(N):
            wm += w[j] * pos[j]
        pts[i] = wm
        va = 0.0
        for j in range(N):
            va += w[j] * (pos[j] - wm) ** 2
        std_[i] = va ** 0.5
        pm = md_v[i]
        pz = z_v[i]
    return pts, std_


# Warm-up JIT
_md = np.linspace(1, 50, 20, np.float64)
_z = np.zeros(20, np.float64)
_gr = np.full(20, 50.0, np.float64)
_gg = np.linspace(45, 55, 100, np.float64)
_pf_ancc(_md, _z, _gr, _gg, 45.0, 0.1, 20.0, 50.0, 0.0, 8,
         0.998, 0.002, 0.005, 0.3, 0.1, 0.001, 0.5)
_pf_z(_md, _z, _gr, _gr, _gg, _gg, 45.0, 0.1, 20.0, 50.0, 0.0,
      -1.0, 0.0, 0.1, 8, 0.993, 0.005, 0.01, 0.3, 0.2, 0.003, 0.5)
_beam_jit(np.random.randn(30), np.random.randn(50), 25, 8, 15.0, 100.0)
_q = np.random.randn(40)
_r = np.random.randn(50)
_dtw_sakoe_chiba(_q, _r, 10)

# =============================================================================
# Python Feature Engineering Functions
# =============================================================================

def _grid(tw_tvt, tw_gr, step=0.2):
    tmin = float(tw_tvt.min())
    tmax = float(tw_tvt.max())
    tvt_g = np.arange(tmin, tmax + step, step)
    return np.interp(tvt_g, tw_tvt, tw_gr).astype(np.float64), float(tmin), float(step)


def _gr_sig(hw, tw_tvt, tw_gr):
    kn = hw[hw["TVT_input"].notna() & hw["GR"].notna()]
    if len(kn) < 20:
        return float(CFG.PF_GR_SIG_DEF)
    return float(np.clip(np.std(kn["GR"].values - np.interp(kn["TVT_input"].values, tw_tvt, tw_gr)),
                         CFG.PF_GR_SIG_MIN, CFG.PF_GR_SIG_MAX))


def _nn(arr, v):
    i = int(np.searchsorted(arr, v, "left"))
    if i >= len(arr):
        return len(arr) - 1
    if i > 0 and abs(arr[i - 1] - v) <= abs(arr[i] - v):
        return i - 1
    return i


def _smooth(vals, fb, r):
    s = pd.Series(vals, dtype="float32").interpolate(limit_direction="both").fillna(fb)
    return (s.rolling(r * 2 + 1, center=True, min_periods=1).mean() if r > 0 else s).to_numpy(np.float32)


def beam_search(gr_h, tw_tvt, tw_gr, start_tvt, bs, mc, es, r):
    si = _nn(tw_tvt, start_tvt)
    sgr = _smooth(gr_h, float(np.nanmean(tw_gr)), r).astype(np.float64)
    path = _beam_jit(sgr, tw_gr.astype(np.float64), si, bs, float(mc), float(es))
    return tw_tvt[path].astype(np.float32)


def run_pf_ancc(hw, tw_tvt, tw_gr, N=CFG.ANCC_N):
    gs = _gr_sig(hw, tw_tvt, tw_gr)
    kn = hw[hw["TVT_input"].notna()]
    ev = hw[hw["TVT_input"].isna()]
    if len(ev) == 0:
        return np.array([]), np.array([])
    ls = float(kn["TVT_input"].iloc[-1] + kn["Z"].iloc[-1])
    tail = kn.tail(30)
    dt = np.diff(tail["TVT_input"].values)
    dz = np.diff(tail["Z"].values)
    dm = np.diff(tail["MD"].values)
    m = dm > 0
    ir = float(np.median((dt + dz)[m] / dm[m])) if m.sum() >= 3 else 0.0
    gg, gmin, gst = _grid(tw_tvt, tw_gr)
    pts, std = _pf_ancc(ev["MD"].values.astype(np.float64), ev["Z"].values.astype(np.float64),
                        ev["GR"].values.astype(np.float64), gg, gmin, gst,
                        gs, ls, ir, N, CFG.ANCC_ALPHA, CFG.ANCC_RN, CFG.ANCC_PN, CFG.ANCC_IS, CFG.ANCC_RP, CFG.ANCC_RR, CFG.PF_RESAMP)
    return pts.astype(np.float32), std.astype(np.float32)


def run_pf_z(hw, tw_tvt, tw_gr, N=CFG.PF_N):
    gs = _gr_sig(hw, tw_tvt, tw_gr)
    tw_s = pd.Series(tw_gr).rolling(CFG.PF_GR_WIN, center=True, min_periods=1).mean().values.astype(np.float32)
    kna = hw[hw["TVT_input"].notna()]
    ev = hw[hw["TVT_input"].isna()]
    if len(ev) == 0:
        return np.array([]), np.array([])
    dz_k = np.diff(kna["Z"].values)
    dvt = np.diff(kna["TVT_input"].values)
    dmd_k = np.diff(kna["MD"].values)
    m2 = dmd_k > 0
    if m2.sum() >= 10:
        vz = dz_k[m2] / dmd_k[m2]
        vt = dvt[m2] / dmd_k[m2]
        A = np.column_stack([vz, np.ones_like(vz)])
        c, _, _, _ = np.linalg.lstsq(A, vt, rcond=None)
        beta, icpt, zsig = float(c[0]), float(c[1]), max(float(np.std(vt - (c[0] * vz + c[1]))), 0.001)
    else:
        beta, icpt, zsig = -1.0, 0.0, 0.1
    t2 = kna.tail(20)
    dvt2 = np.diff(t2["TVT_input"].values)
    dmd2 = np.diff(t2["MD"].values)
    m3 = dmd2 > 0
    iv = float(np.median(dvt2[m3] / dmd2[m3])) if m3.sum() >= 3 else 0.0
    gg, gmin, gst = _grid(tw_tvt, tw_gr)
    gs2, _, _ = _grid(tw_tvt, tw_s)
    gr_sm = hw["GR"].rolling(CFG.PF_GR_WIN, center=True, min_periods=1).mean()
    pts, std = _pf_z(ev["MD"].values.astype(np.float64), ev["Z"].values.astype(np.float64),
                     ev["GR"].values.astype(np.float64),
                     gr_sm.loc[ev.index].values.astype(np.float64),
                     gg, gs2, gmin, gst, gs, float(kna["TVT_input"].iloc[-1]), iv,
                     beta, icpt, zsig, N,
                     CFG.PF_MOM, CFG.PF_VN, CFG.PF_PN, CFG.PF_GR_WT, CFG.PF_ROUGH_P, CFG.PF_ROUGH_V, CFG.PF_RESAMP)
    return pts.astype(np.float32), std.astype(np.float32)


def run_dtw_multiscale(query_gr, tw_tvt, tw_gr, last_tvt, radii=CFG.DTW_RADII):
    N = len(query_gr)
    qn = (query_gr - query_gr.mean()) / (query_gr.std() + 1e-6)
    rn = (tw_gr - tw_gr.mean()) / (tw_gr.std() + 1e-6)
    si = _nn(tw_tvt, last_tvt)
    qn_f = qn.astype(np.float64)
    rn_f = rn.astype(np.float64)
    dtw_tvts = {}
    dtw_costs = {}
    inv_cost_sum = 0.0
    tvt_stack = []
    for r in radii:
        D, pi, pj = _dtw_sakoe_chiba(qn_f, rn_f, r)
        cost = float(D[len(qn_f) - 1, len(rn_f) - 1]) / max(len(qn_f) + len(rn_f), 1)
        tvt_pred = _dtw_path_to_tvt(pi[::-1], pj[::-1], tw_tvt.astype(np.float32), N)
        dtw_tvts[r] = tvt_pred
        dtw_costs[r] = cost
        ic = 1.0 / (cost + 1e-6)
        inv_cost_sum += ic
        tvt_stack.append((tvt_pred, ic))
    weights = np.array([ic / inv_cost_sum for _, ic in tvt_stack], dtype=np.float32)
    tvts_mat = np.stack([t for t, _ in tvt_stack], axis=1)
    dtw_ens = (tvts_mat * weights[None, :]).sum(axis=1).astype(np.float32)
    return dtw_tvts, dtw_costs, dtw_ens


def robust_slope(x, y, w=None):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2 or np.std(x[m]) < 1e-6:
        return 0.0
    return float(np.polyfit(x[m], y[m], 1)[0])


def seg_b_well(ktvt, kz, form_col):
    bv = ktvt + kz - form_col
    n = len(bv)
    b_full = float(np.median(bv))
    return b_full, 0.0, 0.0, 0.0, 0.0


def multi_scale_ncc(kgr, ktvt, hgr, hws=(8, 15, 25), stride=3):
    out = []
    for hw in hws:
        win = 2 * hw + 1
        nk = len(kgr)
        nh = len(hgr)
        if nk < win + 1 or nh == 0:
            out.append((np.full(nh, ktvt[-1], np.float32), np.zeros(nh, np.float32)))
            continue
        kg = pd.Series(kgr).rolling(5, center=True, min_periods=1).mean().values.astype(np.float32)
        hg = pd.Series(hgr).rolling(5, center=True, min_periods=1).mean().values.astype(np.float32)
        sts = np.arange(0, nk - win + 1, stride, dtype=np.int32)
        M = len(sts)
        if M == 0:
            out.append((np.full(nh, ktvt[-1], np.float32), np.zeros(nh, np.float32)))
            continue
        C = kg[sts[:, None] + np.arange(win, dtype=np.int32)[None, :]].astype(np.float32)
        Cn = (C - C.mean(1, keepdims=True)) / (C.std(1, keepdims=True) + 1e-6)
        hp = np.pad(hg, hw, mode="edge")
        H = hp[np.arange(nh)[:, None] + np.arange(win)[None, :]].astype(np.float32)
        Hn = (H - H.mean(1, keepdims=True)) / (H.std(1, keepdims=True) + 1e-6)
        ncc = Hn @ Cn.T / win
        best = ncc.argmax(1)
        score = ncc.max(1).astype(np.float32)
        out.append((ktvt[np.clip(sts[best] + hw, 0, nk - 1)].astype(np.float32), score))
    tvts = np.stack([o[0] for o in out], 1)
    scores = np.stack([o[1] for o in out], 1)
    sw = np.exp(3.0 * scores)
    sw /= sw.sum(1, keepdims=True) + 1e-9
    sc_ens = (tvts * sw).sum(1).astype(np.float32)
    return out, sc_ens


class FormationPlaneKNN:
    def __init__(self, well_ids, data_dir):
        rows = []
        for wid in well_ids:
            p = data_dir / f"{wid}__horizontal_well.csv"
            try:
                df = pd.read_csv(p, usecols=["X", "Y"] + CFG.FORMATIONS).dropna()
            except Exception:
                continue
            if len(df) == 0:
                continue
            row = {"wid": wid, "x": float(df["X"].median()), "y": float(df["Y"].median())}
            for c in CFG.FORMATIONS:
                row[f"{c}_m"] = float(df[c].median())
            rows.append(row)
        self.df = pd.DataFrame(rows)
        self.wmap = {w: i for i, w in enumerate(self.df["wid"])}
        xy = self.df[["x", "y"]].to_numpy()
        self.scale = np.where(xy.std(0) < 1e-3, 1.0, xy.std(0))
        self.tree = cKDTree(xy / self.scale)
        self.xa = self.df["x"].to_numpy()
        self.ya = self.df["y"].to_numpy()
        self.fa = self.df[[f"{c}_m" for c in CFG.FORMATIONS]].to_numpy(np.float64)

    def impute(self, xy_q, self_wid=None, k=CFG.PLANE_K):
        q = xy_q / self.scale
        nf = min(k + 5, len(self.df))
        dist, idx = self.tree.query(q, k=nf, workers=-1)
        if self_wid in self.wmap:
            dist = np.where(idx == self.wmap[self_wid], np.inf, dist)
        ord_ = np.argpartition(dist, min(k - 1, nf - 1), 1)[:, :k]
        dk = np.take_along_axis(dist, ord_, 1)
        ik = np.take_along_axis(idx, ord_, 1)
        vk = np.isfinite(dk)
        w = np.where(vk, 1.0 / (dk + 1e-3), 0.0).astype(np.float64)
        xn = self.xa[ik]
        yn = self.ya[ik]
        fn = self.fa[ik]
        wx = w * xn
        wy = w * yn
        A = np.zeros((len(q), 3, 3))
        A[:, 0, 0] = (wx * xn).sum(1)
        A[:, 0, 1] = (wx * yn).sum(1)
        A[:, 0, 2] = wx.sum(1)
        A[:, 1, 0] = A[:, 0, 1]
        A[:, 1, 1] = (wy * yn).sum(1)
        A[:, 1, 2] = wy.sum(1)
        A[:, 2, 0] = A[:, 0, 2]
        A[:, 2, 1] = A[:, 1, 2]
        A[:, 2, 2] = w.sum(1)
        A[:, 0, 0] += 1e-9
        A[:, 1, 1] += 1e-9
        A[:, 2, 2] += 1e-9
        rhs = np.stack([
            (wx[:, :, None] * fn).sum(1),
            (wy[:, :, None] * fn).sum(1),
            (w[:, :, None] * fn).sum(1)
        ], 1)
        try:
            coef = np.linalg.solve(A, rhs)
        except Exception:
            coef = np.zeros((len(q), 3, 6))
            for r in range(len(q)):
                try:
                    coef[r] = np.linalg.pinv(A[r]) @ rhs[r]
                except Exception:
                    pass
        Xq = xy_q[:, 0]
        Yq = xy_q[:, 1]
        pred = (Xq[:, None] * coef[:, 0, :] + Yq[:, None] * coef[:, 1, :] + coef[:, 2, :]).astype(np.float32)
        pred[~vk.any(1)] = self.fa.mean(0)
        return pred, np.where(vk, dk, np.inf).min(1).astype(np.float32)


class DenseANCCImputer:
    def __init__(self, well_ids, data_dir, spw=CFG.DENSE_SPW):
        xs, ys, anccs, wids = [], [], [], []
        for wid in well_ids:
            p = data_dir / f"{wid}__horizontal_well.csv"
            try:
                df = pd.read_csv(p, usecols=["X", "Y", "ANCC"]).dropna()
            except Exception:
                continue
            if len(df) == 0:
                continue
            ix = np.linspace(0, len(df) - 1, min(spw, len(df)), dtype=int)
            s = df.iloc[ix]
            xs.append(s["X"].values)
            ys.append(s["Y"].values)
            anccs.append(s["ANCC"].values)
            wids.extend([wid] * len(s))
        self.xy = np.column_stack([np.concatenate(xs), np.concatenate(ys)])
        self.ancc = np.concatenate(anccs).astype(np.float32)
        self.wids = np.array(wids)
        self.scale = np.where(self.xy.std(0) < 1e-3, 1.0, self.xy.std(0))
        self.tree = cKDTree(self.xy / self.scale)

    def impute(self, xy_q, self_wid=None, k=CFG.DENSE_K, nfetch=5000):
        xy_q = np.atleast_2d(xy_q)
        q = xy_q / self.scale
        nf = min(nfetch, len(self.ancc))
        dist, idx = self.tree.query(q, k=nf, workers=-1)
        if self_wid:
            dist = np.where(self.wids[idx] == self_wid, np.inf, dist)
        ord_ = np.argpartition(dist, min(k - 1, nf - 1), 1)[:, :k]
        dk = np.take_along_axis(dist, ord_, 1)
        ik = np.take_along_axis(idx, ord_, 1)
        vk = np.isfinite(dk)
        w = np.where(vk, 1.0 / (dk + 1e-3), 0.0)
        sw = w.sum(1)
        safe = np.where(sw < 1e-9, 1.0, sw)
        an = self.ancc[ik]
        ap = (an * w).sum(1) / safe
        ap = np.where(sw < 1e-9, float(self.ancc.mean()), ap)
        var = ((an - ap[:, None]) ** 2 * w).sum(1) / safe
        return ap.astype(np.float32), np.sqrt(np.maximum(var, 0.0)).astype(np.float32), np.where(vk, dk, np.inf).min(1).astype(np.float32)


# =============================================================================
# STTRE Model (Must match training exactly)
# =============================================================================

class HSPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].size(1)])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(-2)]
        return self.dropout(x)


class RelativeMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5
        self.qkv_linear = nn.Linear(d_model, d_model * 3)
        self.fc_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask, rel_bias):
        B, L, _ = x.shape
        qkv = self.qkv_linear(x).chunk(3, dim=-1)
        q = qkv[0].view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        k = qkv[1].view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        v = qkv[2].view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale + rel_bias
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(mask == True, float("-inf"))
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        context = torch.matmul(attn_probs, v)
        context = context.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.fc_out(context)


class RelativeTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dim_feedforward, dropout):
        super().__init__()
        self.self_attn = RelativeMultiHeadAttention(d_model, n_heads, dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, src, src_key_padding_mask, rel_bias):
        src2 = self.self_attn(src, src_key_padding_mask, rel_bias)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class RelativeTransformerEncoder(nn.Module):
    def __init__(self, layer, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_key_padding_mask, rel_bias):
        output = src
        for mod in self.layers:
            output = mod(output, src_key_padding_mask, rel_bias)
        if self.norm is not None:
            output = self.norm(output)
        return output


class STTRE_Rogii(nn.Module):
    def __init__(self, input_dim, d_model, n_heads, num_encoder_layers, dropout,
                 n_estimators, max_seq_len, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.n_estimators = n_estimators
        self.max_seq_len = max_seq_len

        self.feature_embed = nn.Linear(input_dim, d_model)
        self.role_embed = nn.Embedding(n_estimators + 1, d_model, padding_idx=n_estimators)
        self.time_embed = HSPositionalEncoding(d_model, dropout, max_len=max_seq_len)
        self.rel_time_bias = nn.Embedding(2 * max_seq_len - 1, n_heads)
        self.rel_role_bias = nn.Embedding((n_estimators + 1) * (n_estimators + 1), n_heads)
        self.attn_pool_layer = nn.Linear(d_model, 1)

        common_ff_dim = d_model * 4
        rel_time_layer = RelativeTransformerEncoderLayer(d_model, n_heads, common_ff_dim, dropout)
        self.temporal_branch = RelativeTransformerEncoder(rel_time_layer, num_encoder_layers, nn.LayerNorm(d_model))
        rel_role_layer = RelativeTransformerEncoderLayer(d_model, n_heads, common_ff_dim, dropout)
        self.spatial_branch = RelativeTransformerEncoder(rel_role_layer, num_encoder_layers, nn.LayerNorm(d_model))
        rel_time_layer_2 = RelativeTransformerEncoderLayer(d_model, n_heads, common_ff_dim, dropout)
        self.spatio_temporal_branch = RelativeTransformerEncoder(rel_time_layer_2, num_encoder_layers, nn.LayerNorm(d_model))
        self.pool_projection = nn.Linear(d_model * 2, d_model)

        self.cross_attn_spatial_to_temporal = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm_cross_attn_S = nn.LayerNorm(d_model)
        self.dropout_cross_attn_S = nn.Dropout(dropout)
        self.cross_attn_temporal_to_spatial = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm_cross_attn_T = nn.LayerNorm(d_model)
        self.dropout_cross_attn_T = nn.Dropout(dropout)

        self.fusion_gate_mlp = nn.Sequential(
            nn.Linear(d_model * 3, d_model * 2), nn.ReLU(),
            nn.Linear(d_model * 2, d_model * 3), nn.Sigmoid()
        )
        self.final_feedforward = nn.Sequential(
            nn.LayerNorm(d_model * 3),
            nn.Linear(d_model * 3, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout * 2)
        )

        # =============================================================================
        # IMPROVED: Enhanced per-time-step prediction head
        # =============================================================================
        # 1. Temporal convolution for local smoothness
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(d_model * 3, d_model * 2, kernel_size=5, padding=2, groups=1),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 2. Multi-scale temporal fusion (different receptive fields)
        self.temporal_conv_short = nn.Sequential(
            nn.Conv1d(d_model, d_model // 2, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.temporal_conv_long = nn.Sequential(
            nn.Conv1d(d_model, d_model // 2, kernel_size=7, padding=3),
            nn.GELU(),
        )

        # 3. Combine global well info + multi-scale local info
        self.temporal_fusion = nn.Sequential(
            nn.LayerNorm(d_model * 4),
            nn.Linear(d_model * 4, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 4. Final regression with residual from spatio-temporal
        self.regression_head = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model, 1),
        )

    def _get_relative_time_bias(self, T, device):
        q_indices = torch.arange(T, device=device)
        k_indices = torch.arange(T, device=device)
        rel_indices = k_indices.unsqueeze(0) - q_indices.unsqueeze(1) + (T - 1)
        rel_indices = torch.clamp(rel_indices, 0, 2 * self.max_seq_len - 2)
        rel_bias = self.rel_time_bias(rel_indices)
        return rel_bias.permute(2, 0, 1).unsqueeze(0)

    def _get_relative_role_bias(self, role_ids, device):
        B, N = role_ids.shape
        q_roles = role_ids.unsqueeze(2).expand(-1, -1, N)
        k_roles = role_ids.unsqueeze(1).expand(-1, N, -1)
        rel_role_indices = q_roles * (self.n_estimators + 1) + k_roles
        rel_bias = self.rel_role_bias(rel_role_indices)
        return rel_bias.permute(0, 3, 1, 2)

    def forward(self, x, role_ids, src_key_padding_mask):
        B, N_max, T, F = x.shape

        x_emb = self.feature_embed(x)
        roles_emb = self.role_embed(role_ids)
        roles_emb_expanded = roles_emb.unsqueeze(2).expand(-1, -1, T, -1)

        input_temp_st = (x_emb + roles_emb_expanded).view(B * N_max, T, self.d_model)
        input_temp_st = self.time_embed(input_temp_st)
        pooled_spatial_input = (x_emb + roles_emb_expanded)[:, :, -1, :]

        rel_time_bias = self._get_relative_time_bias(T, x.device)
        rel_role_bias = self._get_relative_role_bias(role_ids, x.device)
        rel_time_bias_expanded = rel_time_bias.expand(B * N_max, -1, -1, -1)

        out_temporal = self.temporal_branch(input_temp_st, None, rel_time_bias_expanded)
        out_spatial = self.spatial_branch(pooled_spatial_input, src_key_padding_mask, rel_role_bias)
        out_spatio_temporal = self.spatio_temporal_branch(input_temp_st, None, rel_time_bias_expanded)

        pooled_temporal = out_temporal[:, -1, :].view(B, N_max, self.d_model)

        attn_scores = self.attn_pool_layer(out_spatio_temporal)
        attn_weights = torch.softmax(attn_scores, dim=1)
        st_attn = torch.sum(out_spatio_temporal * attn_weights, dim=1)
        st_last = out_spatio_temporal[:, -1, :]
        st_mixed = torch.cat([st_attn, st_last], dim=-1)
        st_projected = self.pool_projection(st_mixed).view(B, N_max, self.d_model)

        target_temporal_original = pooled_temporal[:, 0, :]
        target_spatial_original = out_spatial[:, 0, :]
        target_spatio_temporal_mixed = st_projected[:, 0, :]

        query_S = target_spatial_original.unsqueeze(1)
        cross_attn_output_S, _ = self.cross_attn_spatial_to_temporal(
            query=query_S, key=pooled_temporal, value=pooled_temporal,
            key_padding_mask=src_key_padding_mask
        )
        target_spatial_fused = self.norm_cross_attn_S(
            query_S + self.dropout_cross_attn_S(cross_attn_output_S)
        ).squeeze(1)

        query_T = target_temporal_original.unsqueeze(1)
        cross_attn_output_T, _ = self.cross_attn_temporal_to_spatial(
            query=query_T, key=out_spatial, value=out_spatial,
            key_padding_mask=src_key_padding_mask
        )
        target_temporal_fused = self.norm_cross_attn_T(
            query_T + self.dropout_cross_attn_T(cross_attn_output_T)
        ).squeeze(1)

        h_target_combined_input = torch.cat([
            target_temporal_fused, target_spatial_fused, target_spatio_temporal_mixed
        ], dim=1)
        gate_weights = self.fusion_gate_mlp(h_target_combined_input)
        h_target_combined_gated = h_target_combined_input * gate_weights

        h_final = self.final_feedforward(h_target_combined_gated)

        # =============================================================================
        # IMPROVED: Per-time-step prediction with enhanced head
        # =============================================================================
        out_st = out_spatio_temporal.view(B, N_max, T, self.d_model)
        mask_expanded = src_key_padding_mask.unsqueeze(-1).unsqueeze(-1)
        out_st_masked = out_st.masked_fill(mask_expanded, 0.0)
        valid_count = (~src_key_padding_mask).sum(dim=1, keepdim=True).unsqueeze(-1).float()
        st_local = out_st_masked.sum(dim=1) / (valid_count + 1e-8)  # (B, T, d_model)

        # Global well representation expanded to each timestep
        h_well = h_final.unsqueeze(1).expand(-1, T, -1)  # (B, T, d_model*2)

        # Multi-scale temporal conv on local features
        # st_local: (B, T, d_model) -> (B, d_model, T) for conv1d
        st_local_t = st_local.permute(0, 2, 1)  # (B, d_model, T)
        st_short = self.temporal_conv_short(st_local_t).permute(0, 2, 1)  # (B, T, d_model//2)
        st_long = self.temporal_conv_long(st_local_t).permute(0, 2, 1)    # (B, T, d_model//2)
        st_multi = torch.cat([st_short, st_long], dim=-1)  # (B, T, d_model)

        # Combine: global + local + multi-scale
        h_combined = torch.cat([h_well, st_local, st_multi], dim=-1)  # (B, T, d_model*4)
        h_fused = self.temporal_fusion(h_combined)  # (B, T, d_model*2)

        # Final prediction
        pred = self.regression_head(h_fused).squeeze(-1)  # (B, T)

        return pred


# =============================================================================
# Build Well Features for Inference
# =============================================================================

def build_well_infer(hw_path, tw_path, fi, di):
    wid = Path(hw_path).stem.replace("__horizontal_well", "")
    hw = pd.read_csv(hw_path)
    tw = pd.read_csv(tw_path).sort_values("TVT")
    kn = hw[hw["TVT_input"].notna()]
    ev = hw[hw["TVT_input"].isna()]
    if len(ev) == 0 or len(kn) < 10:
        return None
    tw_tvt = tw["TVT"].to_numpy(np.float32)
    tw_gr = tw["GR"].to_numpy(np.float32)
    if len(tw_tvt) < 3:
        return None

    pf_a, std_a = run_pf_ancc(hw, tw_tvt, tw_gr)
    if len(pf_a) == 0:
        return None
    pf_z, std_z = run_pf_z(hw, tw_tvt, tw_gr)
    pf_use = pf_a.astype(np.float32)
    has_z = len(pf_z) == len(pf_a) and not np.any(np.isnan(pf_z))

    lk = kn.iloc[-1]
    last_tvt = float(lk["TVT_input"])
    gr_full = hw["GR"].astype(float).interpolate(limit_direction="both").fillna(float(np.nanmean(tw_gr)))
    hgr = gr_full.iloc[ev.index[0]:].to_numpy(np.float32)
    kgr = gr_full.iloc[:len(kn)].to_numpy(np.float32)

    bpaths = {}
    for (bs, mc, es, r, tag) in CFG.BEAMS:
        bpaths[tag] = beam_search(hgr, tw_tvt, tw_gr, last_tvt, bs, mc, es, r)

    ktvt = kn["TVT_input"].to_numpy(np.float32)
    sc_res, sc_ens = multi_scale_ncc(kgr, ktvt, hgr, hws=(8, 15, 25), stride=3)
    sc8, _ = sc_res[0]
    sc15, _ = sc_res[1]
    sc25, _ = sc_res[2]

    full_gr = gr_full.values.astype(np.float32)
    dtw_tvts_ms, dtw_costs_ms, dtw_ens_ms = run_dtw_multiscale(full_gr, tw_tvt, tw_gr, last_tvt, radii=CFG.DTW_RADII)

    nh = len(ev)
    ev_start = ev.index[0]

    def _ev_slice(arr):
        return arr[ev_start:ev_start + nh].astype(np.float32)

    dtw_ens_ev = _ev_slice(dtw_ens_ms)

    # Z-prior baseline
    full_z = hw["Z"].values.astype(np.float64)
    full_dz = np.diff(full_z, prepend=full_z[0])
    ev_dz = full_dz[ev.index].astype(np.float32)
    dtvt_from_dz = -ev_dz
    z_prior = np.cumsum(dtvt_from_dz).astype(np.float32)
    z_baseline = np.float32(last_tvt) + z_prior

    # DTW cost feature
    inv_cost_sum = sum(1.0 / (dtw_costs_ms[r] + 1e-6) for r in CFG.DTW_RADII)
    dtw_cost = np.full(nh, 0.0, np.float32)
    for r in CFG.DTW_RADII:
        w = (1.0 / (dtw_costs_ms[r] + 1e-6)) / inv_cost_sum
        dtw_cost += float(w)

    xy_ev = ev[["X", "Y"]].to_numpy(np.float64)
    xy_kn = kn[["X", "Y"]].to_numpy(np.float64)
    form_ev, knn_d = fi.impute(xy_ev, self_wid=None)
    form_kn, _ = fi.impute(xy_kn, self_wid=None)
    z_kn = kn["Z"].to_numpy(np.float32)
    z_ev = ev["Z"].to_numpy(np.float32)

    tvt_fs = {}
    for fi2, fn in enumerate(CFG.FORMATIONS):
        b_full, _, _, _, _ = seg_b_well(ktvt, z_kn, form_kn[:, fi2])
        tvt_f = (-z_ev + form_ev[:, fi2] + b_full).astype(np.float32)
        tvt_fs[f"tvtF_{fn}"] = tvt_f

    d_ancc, _, _ = di.impute(xy_ev, self_wid=None)
    d_kn, _, _ = di.impute(xy_kn, self_wid=None)
    b_vd = ktvt + z_kn - d_kn
    b_d = float(np.median(b_vd))
    tvt_dense = (-z_ev + d_ancc + b_d).astype(np.float32)

    estimator_tvts = {
        "pf_ancc": pf_use,
        "pf_z": pf_z.astype(np.float32) if has_z else np.full(nh, last_tvt, np.float32),
        "beam_cons": bpaths["cons"], "beam_sm5": bpaths["sm5"],
        "sc_ens": sc_ens, "dtw_ens": dtw_ens_ev,
        "tvtF_ANCC": tvt_fs["tvtF_ANCC"], "tvt_dense": tvt_dense,
    }

    hmd = ev["MD"].to_numpy(np.float32)
    md_since = hmd - float(lk["MD"])
    frac = (np.arange(nh) / max(nh - 1, 1)).astype(np.float32)
    mdd = hw["MD"].diff().replace(0, np.nan)
    dzdmd = (hw["Z"].diff() / mdd).iloc[ev.index].values.astype(np.float32)
    dxdmd = (hw["X"].diff() / mdd).iloc[ev.index].values.astype(np.float32)
    dydmd = (hw["Y"].diff() / mdd).iloc[ev.index].values.astype(np.float32)
    gr_s = pd.Series(gr_full.values)
    gr_d1 = gr_s.diff().fillna(0.0).iloc[ev.index].values.astype(np.float32)
    gr_d2 = gr_s.diff().diff().fillna(0.0).iloc[ev.index].values.astype(np.float32)
    gr_env = gr_s.rolling(21, center=True, min_periods=1).max().iloc[ev.index].values.astype(np.float32)
    gr_nrg = np.sqrt(np.maximum((gr_s ** 2).rolling(21, center=True, min_periods=1).mean(), 0.0)).iloc[ev.index].values.astype(np.float32)

    pf_ancc_std = std_a[:nh].astype(np.float32) if len(std_a) >= nh else np.full(nh, float(np.nanstd(std_a) if len(std_a) > 0 else 0.0), np.float32)
    pf_z_std = std_z[:nh].astype(np.float32) if (has_z and len(std_z) >= nh) else np.full(nh, float(np.nanstd(std_z) if len(std_z) > 0 else 0.0), np.float32)

    n_est = len(CFG.ESTIMATORS)
    base_feats = {
        "gr": hgr, "z": z_ev, "x": ev["X"].to_numpy(np.float32), "y": ev["Y"].to_numpy(np.float32),
        "md": hmd, "md_since": md_since, "frac": frac,
        "dzdmd": dzdmd, "dxdmd": dxdmd, "dydmd": dydmd,
        "gr_d1": gr_d1, "gr_d2": gr_d2, "gr_env": gr_env, "gr_nrg": gr_nrg,
        "z_prior": z_prior, "dtvt_from_dz": dtvt_from_dz,
        "pf_ancc_std": pf_ancc_std, "pf_z_std": pf_z_std,
        "dtw_cost": dtw_cost, "knn_d": knn_d[:nh].astype(np.float32),
    }
    F = len(base_feats) + 2

    seq = np.zeros((n_est, nh, F), dtype=np.float32)
    for est_idx, est_name in enumerate(CFG.ESTIMATORS):
        est_tvt = estimator_tvts[est_name]
        est_delta = est_tvt - last_tvt
        seq[est_idx, :, 0] = est_tvt
        seq[est_idx, :, 1] = est_delta
        for f_idx, (_, feat_vals) in enumerate(base_feats.items(), start=2):
            seq[est_idx, :, f_idx] = feat_vals

    ids = [f"{wid}_{i}" for i in ev.index]
    return seq, last_tvt, ids, z_baseline


# =============================================================================
# Model Loading & Sliding Window Prediction
# =============================================================================

def load_models():
    # Try common Kaggle dataset mount paths; also search recursively for config file
    search_roots = [
        Path("/kaggle/input/rogii-sttre-v6-fullwellcv"),
        Path("/kaggle/input/datasets/jezzlin/rogii-sttre-v6-fullwellcv"),
        CFG.MODEL_DIR,
        CFG.MODEL_DIR.parent,
    ]
    config_name = "model_config_v6_fullwellcv.json"
    scaler_name = "scaler_v6_fullwellcv.npz"
    model_dir = None
    for root in search_roots:
        if not root.exists():
            continue
        direct = root / "sttre_rogii_v6_fullwellcv"
        if direct.exists() and (direct / config_name).exists():
            model_dir = direct
            break
        # Search one level deeper
        for cand in root.rglob(config_name):
            model_dir = cand.parent
            break
        if model_dir is not None:
            break

    if model_dir is None:
        # List searchable paths for diagnostics
        diag = []
        for root in search_roots:
            if root.exists():
                try:
                    diag.append(f"{root}: {list(root.iterdir())}")
                except Exception as e:
                    diag.append(f"{root}: error {e}")
            else:
                diag.append(f"{root}: does not exist")
        raise FileNotFoundError(f"Model directory not found. Searched: {search_roots}. Diagnostics: {diag}")
    print(f"Loading models from {model_dir}")

    with open(model_dir / config_name, "r") as f:
        config = json.load(f)
    scaler = np.load(model_dir / scaler_name)
    scaler_mean = scaler["mean"].astype(np.float32)
    scaler_scale = scaler["scale"].astype(np.float32)

    models = []
    for fold_idx in range(5):
        model_path = model_dir / f"sttre_fold{fold_idx}_best.pth"
        if not model_path.exists():
            print(f"Warning: {model_path} not found, skipping fold {fold_idx}")
            continue
        model = STTRE_Rogii(**config)
        checkpoint = torch.load(model_path, map_location=CFG.DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(CFG.DEVICE).eval()
        models.append(model)
        print(f"Loaded fold {fold_idx}")
    return models, scaler_mean, scaler_scale


def predict_single(model, seq, scaler_mean, scaler_scale):
    N, T, F = seq.shape
    seq_flat = seq.reshape(-1, F)
    seq_scaled = (seq_flat - scaler_mean) / (scaler_scale + 1e-8)
    seq_scaled = seq_scaled.reshape(N, T, F)

    seq_tensor = torch.tensor(seq_scaled, dtype=torch.float32).unsqueeze(0).to(CFG.DEVICE)
    role_ids = torch.arange(N, device=CFG.DEVICE).unsqueeze(0)
    mask = torch.ones(1, N, dtype=torch.bool, device=CFG.DEVICE)

    with torch.no_grad():
        pred = model(seq_tensor, role_ids, ~mask)

    return pred.squeeze(0).cpu().numpy()


def predict_sliding_window(models, seq, z_baseline, scaler_mean, scaler_scale):
    N, T, F = seq.shape
    W = CFG.MAX_SEQ_LEN
    S = W // 2

    if T <= W:
        preds = []
        for model in models:
            preds.append(predict_single(model, seq, scaler_mean, scaler_scale))
        pred_residual = np.mean(preds, axis=0)
        return (pred_residual + z_baseline).astype(np.float32)

    all_preds = np.zeros(T, dtype=np.float64)
    all_counts = np.zeros(T, dtype=np.float64)

    starts = list(range(0, T - W + 1, S))
    if not starts or starts[-1] + W < T:
        starts.append(T - W)

    for start in starts:
        end = min(start + W, T)
        window_seq = seq[:, start:end, :].copy()
        actual_len = end - start

        if actual_len < W:
            pad = W - actual_len
            window_seq = np.concatenate([window_seq, np.zeros((N, pad, F), dtype=np.float32)], axis=1)

        preds = []
        for model in models:
            preds.append(predict_single(model, window_seq, scaler_mean, scaler_scale))
        pred_residual = np.mean(preds, axis=0)
        pred_residual = pred_residual[:actual_len]
        pred_tvt = pred_residual + z_baseline[start:end]

        all_preds[start:end] += pred_tvt
        all_counts[start:end] += 1

    final_pred = all_preds / np.maximum(all_counts, 1)
    return final_pred.astype(np.float32)


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("STTRE-Rogii v6 Full-Well CV Submission")
    print("=" * 60)

    print("\nLoading models...")
    models, scaler_mean, scaler_scale = load_models()
    if len(models) == 0:
        print("No models found!")
        return
    print(f"Loaded {len(models)} models")

    print("\nLoading spatial imputers...")
    train_hw_paths = sorted((CFG.DATA_DIR / "train").glob("*__horizontal_well.csv"))
    train_wids = [p.stem.replace("__horizontal_well", "") for p in train_hw_paths]
    fi = FormationPlaneKNN(train_wids, CFG.DATA_DIR / "train")
    di = DenseANCCImputer(train_wids, CFG.DATA_DIR / "train")
    print(f"Imputers ready ({len(train_wids)} train wells)")

    print("\nProcessing test wells...")
    test_hw_paths = sorted((CFG.DATA_DIR / "test").glob("*__horizontal_well.csv"))
    print(f"Found {len(test_hw_paths)} test wells")

    all_ids = []
    all_preds = []

    for i, hw_path in enumerate(test_hw_paths):
        tw_path = hw_path.parent / f"{hw_path.stem.replace('__horizontal_well', '')}__typewell.csv"
        if not tw_path.exists():
            continue

        result = build_well_infer(str(hw_path), str(tw_path), fi, di)
        if result is None:
            continue

        seq, last_tvt, ids, z_baseline = result
        pred_tvt = predict_sliding_window(models, seq, z_baseline, scaler_mean, scaler_scale)

        all_ids.extend(ids)
        all_preds.extend(pred_tvt)

        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(test_hw_paths)} wells")

    print(f"\nSaving submission: {CFG.OUTPUT_PATH}")
    sub = pd.DataFrame({"id": all_ids, "tvt": all_preds})
    sub.to_csv(CFG.OUTPUT_PATH, index=False)
    print(f"Submission saved: {len(sub)} rows")
    print(sub.head())
    print("...")
    print(sub.tail())


if __name__ == "__main__":
    main()

# --- cell 1 ---
