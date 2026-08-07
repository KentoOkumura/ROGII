# ruff: noqa
from __future__ import annotations

import json
import hashlib
import multiprocessing
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numba import njit
from scipy.signal import savgol_filter
from scipy.spatial import cKDTree
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold


class CFG:
    DATA = Path(os.environ.get("ROGII_DATA", "data/raw"))
    OUT = Path(os.environ.get("ROGII_OUT", "."))
    seed = 42
    n_splits = 5
    n_jobs = min(8, multiprocessing.cpu_count())
    PF_SEEDS = 128
    PF_PARTICLES = 500
    PF_SCALES = (3.0, 5.0, 8.0, 12.0)
    FAST = bool(int(os.environ.get("FAST", "0")))
    N_TRAIN_WELLS = int(os.environ.get("N_TRAIN_WELLS", "0"))
    USE_GPU = os.environ.get("USE_GPU", "auto")
    SHOW_FIGS = False


FORMATIONS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]


def stable_seed(*parts: object, modulo: int = 2_147_483_647) -> int:
    key = "::".join(str(part) for part in parts)
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % modulo + 1


def configure_public_runtime(
    *,
    data_dir: str | Path,
    output_dir: str | Path,
    n_jobs: int | None = None,
    pf_seeds: int | None = None,
    pf_particles: int | None = None,
    fast: bool | None = None,
    use_gpu: str | None = None,
    n_train_wells: int | None = None,
) -> None:
    CFG.DATA = Path(data_dir)
    CFG.OUT = Path(output_dir)
    if n_jobs is not None:
        CFG.n_jobs = int(n_jobs)
    if pf_seeds is not None:
        CFG.PF_SEEDS = int(pf_seeds)
    if pf_particles is not None:
        CFG.PF_PARTICLES = int(pf_particles)
    if fast is not None:
        CFG.FAST = bool(fast)
    if use_gpu is not None:
        CFG.USE_GPU = str(use_gpu)
    if n_train_wells is not None:
        CFG.N_TRAIN_WELLS = int(n_train_wells)


def load_well(wid, split="train"):
    base = CFG.DATA / split
    hw = pd.read_csv(base / f"{wid}__horizontal_well.csv")
    tw = pd.read_csv(base / f"{wid}__typewell.csv").sort_values("TVT")
    return hw, tw


def rmse(a, b):
    return float(np.sqrt(mean_squared_error(np.asarray(a, float), np.asarray(b, float))))

PF_N = 600; ANCC_N = 600
PF_MOM = 0.993; PF_VN = 0.005; PF_PN = 0.01
PF_GR_SIG_MIN = 10.; PF_GR_SIG_MAX = 60.; PF_GR_SIG_DEF = 30.
PF_GR_WIN = 5; PF_GR_WT = 0.3; PF_RESAMP = 0.5; PF_ROUGH_P = 0.2; PF_ROUGH_V = 0.003
ANCC_ALPHA = 0.998; ANCC_RN = 0.002; ANCC_PN = 0.005; ANCC_IS = 0.3; ANCC_RP = 0.1; ANCC_RR = 0.001

BEAMS = [(10,20.,144.,2,"cons"),(10,8.,64.,2,"loose"),(8,35.,220.,1,"vcons"),
         (10,14.,90.,5,"sm5"),(20,4.,36.,3,"vloose"),(12,12.,100.,3,"mid"),(15,25.,180.,2,"stiff")]

@njit(cache=True)
def _interp1(grid, v, vmin, step):
    i = int((v - vmin) / step)
    if i < 0: return grid[0]
    n = len(grid) - 1
    if i >= n: return grid[n]
    t = (v - vmin) / step - i
    return grid[i]*(1.-t) + grid[i+1]*t

@njit(cache=True)
def _resamp(pos, aux, w, N, rp, rv):
    cum = np.zeros(N+1)
    for j in range(N): cum[j+1] = cum[j]+w[j]
    u0 = np.random.uniform(0., 1./N); np2 = np.empty(N); na = np.empty(N); ci = 0
    for j in range(N):
        u = u0+j/N
        while ci < N-1 and cum[ci+1] < u: ci += 1
        np2[j] = pos[ci]+rp*np.random.randn(); na[j] = aux[ci]+rv*np.random.randn()
    return np2, na

@njit(cache=True)
def _beam_jit(sgr, tw_gr, si, BS, mc, es):
    n = len(sgr); nt = len(tw_gr); MAX = BS*6
    bidx = np.zeros(BS, np.int64); bidx[0] = si
    bcost = np.full(BS, 1e30); bcost[0] = 0.; bn = np.int64(1)
    hI = np.zeros((n, BS), np.int64); hP = np.zeros((n, BS), np.int64)
    cI = np.zeros(MAX, np.int64); cC = np.full(MAX, 1e30); cP = np.zeros(MAX, np.int64)
    for step in range(n):
        gv = sgr[step]; nc = np.int64(0)
        for bi in range(bn):
            idx = bidx[bi]; cost = bcost[bi]
            for d in range(-2, 3):
                ni = idx+d
                if ni < 0 or ni >= nt: continue
                tot = cost+(gv-tw_gr[ni])**2/es+mc*(d if d >= 0 else -d)
                fnd = np.int64(-1)
                for ci in range(nc):
                    if cI[ci] == ni: fnd = ci; break
                if fnd >= 0:
                    if tot < cC[fnd]: cC[fnd] = tot; cP[fnd] = bi
                else:
                    if nc < MAX: cI[nc] = ni; cC[nc] = tot; cP[nc] = bi; nc += 1
        kept = min(BS, nc)
        for i in range(kept):
            mi = i
            for j in range(i+1, nc):
                if cC[j] < cC[mi]: mi = j
            if mi != i:
                cI[i], cI[mi] = cI[mi], cI[i]; cC[i], cC[mi] = cC[mi], cC[i]; cP[i], cP[mi] = cP[mi], cP[i]
        hI[step, :kept] = cI[:kept]; hP[step, :kept] = cP[:kept]
        bidx[:kept] = cI[:kept]; bcost[:kept] = cC[:kept]; bn = kept
    best = np.int64(0)
    for b in range(1, bn):
        if bcost[b] < bcost[best]: best = b
    path = np.zeros(n, np.int64); b = best
    for s in range(n-1, -1, -1): path[s] = hI[s, b]; b = hP[s, b]
    return path

@njit(cache=True)
def _pf_ancc(md_v, z_v, gr_v, gg, vmin, step, gs, ls, ir, N, ALPHA, RN, PN, IS, RP, RR, RESAMP):
    pos = np.empty(N); rate = np.empty(N); w = np.ones(N)/N
    for j in range(N):
        pos[j] = ls+IS*np.random.randn(); rate[j] = ir+0.01*np.random.randn()
    pts = np.empty(len(md_v)); std_ = np.empty(len(md_v)); pm = md_v[0]-1.
    for i in range(len(md_v)):
        dm = md_v[i]-pm; dm = max(dm, 1.)
        for j in range(N):
            rate[j] = ALPHA*rate[j]+RN*np.random.randn(); pos[j] += rate[j]*dm+PN*np.random.randn()
            tvt_j = pos[j]-z_v[i]; tvt_j = max(tvt_j, vmin-50.); tvt_j = min(tvt_j, vmin+len(gg)*step+50.)
            pos[j] = tvt_j+z_v[i]
        if not np.isnan(gr_v[i]):
            ws = 0.
            for j in range(N):
                eg = _interp1(gg, pos[j]-z_v[i], vmin, step); d = (gr_v[i]-eg)/gs
                lk = max(np.exp(-0.5*d*d) if d*d < 600. else 0., 1e-300); w[j] *= lk; ws += w[j]
            if ws > 0.:
                for j in range(N): w[j] /= ws
            else:
                for j in range(N): w[j] = 1./N
        ne = 0.
        for j in range(N): ne += w[j]*w[j]
        if 1./ne < RESAMP*N:
            pos, rate = _resamp(pos, rate, w, N, RP, RR)
            for j in range(N): w[j] = 1./N
        tv = 0.
        for j in range(N): tv += w[j]*(pos[j]-z_v[i])
        pts[i] = tv; va = 0.
        for j in range(N): va += w[j]*(pos[j]-z_v[i]-tv)**2
        std_[i] = va**0.5; pm = md_v[i]
    return pts, std_

@njit(cache=True)
def _pf_z(md_v, z_v, gr_v, gr_sm_v, gg_p, gg_s, vmin, step, gs, ip, iv, beta, icpt, zsig, N,
         MOM, VN, PN, GR_WT, RP, RV, RESAMP):
    pos = np.empty(N); vel = np.empty(N); w = np.ones(N)/N
    for j in range(N):
        pos[j] = ip+0.5*np.random.randn(); vel[j] = iv+0.02*np.random.randn()
    pts = np.empty(len(md_v)); std_ = np.empty(len(md_v)); pm = md_v[0]-1.; pz = z_v[0]-1.
    for i in range(len(md_v)):
        dm = md_v[i]-pm; dm = max(dm, 1.); dzd = (z_v[i]-pz)/dm; ve = beta*dzd+icpt
        for j in range(N):
            vel[j] = MOM*vel[j]+VN*np.random.randn(); pos[j] += vel[j]*dm+PN*np.random.randn()
            pos[j] = max(pos[j], vmin-50.); pos[j] = min(pos[j], vmin+len(gg_p)*step+50.)
        if not np.isnan(gr_v[i]):
            ws = 0.
            for j in range(N):
                ep = _interp1(gg_p, pos[j], vmin, step); dp = (gr_v[i]-ep)/gs
                lp = max(np.exp(-0.5*dp*dp) if dp*dp < 600. else 0., 1e-300)
                if not np.isnan(gr_sm_v[i]):
                    es = _interp1(gg_s, pos[j], vmin, step); ds = (gr_sm_v[i]-es)/(gs*1.5)
                    lsm = max(np.exp(-0.5*ds*ds) if ds*ds < 600. else 0., 1e-300); lk = (1.-GR_WT)*lp+GR_WT*lsm
                else: lk = lp
                lk = max(lk, 1e-300); w[j] *= lk; ws += w[j]
            if ws > 0.:
                for j in range(N): w[j] /= ws
            else:
                for j in range(N): w[j] = 1./N
        ws2 = 0.
        for j in range(N):
            dv = (vel[j]-ve)/max(zsig*2., 0.005); lz = max(np.exp(-0.5*dv*dv) if dv*dv < 600. else 0., 1e-300)
            w[j] *= lz; ws2 += w[j]
        if ws2 > 0.:
            for j in range(N): w[j] /= ws2
        else:
            for j in range(N): w[j] = 1./N
        ne = 0.
        for j in range(N): ne += w[j]*w[j]
        if 1./ne < RESAMP*N:
            pos, vel = _resamp(pos, vel, w, N, RP, RV)
            for j in range(N): w[j] = 1./N
        wm = 0.
        for j in range(N): wm += w[j]*pos[j]
        pts[i] = wm; va = 0.
        for j in range(N): va += w[j]*(pos[j]-wm)**2
        std_[i] = va**0.5; pm = md_v[i]; pz = z_v[i]
    return pts, std_

def _grid(tw_tvt, tw_gr, step=0.2):
    tmin = float(tw_tvt.min()); tmax = float(tw_tvt.max())
    tvt_g = np.arange(tmin, tmax+step, step)
    return np.interp(tvt_g, tw_tvt, tw_gr).astype(np.float64), float(tmin), float(step)

def _gr_sig(hw, tw_tvt, tw_gr):
    kn = hw[hw.TVT_input.notna() & hw.GR.notna()]
    if len(kn) < 20: return float(PF_GR_SIG_DEF)
    return float(np.clip(np.std(kn.GR.values-np.interp(kn.TVT_input.values, tw_tvt, tw_gr)),
                         PF_GR_SIG_MIN, PF_GR_SIG_MAX))

def _nn(arr, v):
    i = int(np.searchsorted(arr, v, "left"))
    if i >= len(arr): return len(arr)-1
    if i > 0 and abs(arr[i-1]-v) <= abs(arr[i]-v): return i-1
    return i

def _smooth(vals, fb, r):
    s = pd.Series(vals, dtype="float32").interpolate(limit_direction="both").fillna(fb)
    return (s.rolling(r*2+1, center=True, min_periods=1).mean() if r > 0 else s).to_numpy(np.float32)

def beam_search(gr_h, tw_tvt, tw_gr, start_tvt, bs, mc, es, r):
    si = _nn(tw_tvt, start_tvt); sgr = _smooth(gr_h, float(np.nanmean(tw_gr)), r).astype(np.float64)
    return tw_tvt[_beam_jit(sgr, tw_gr.astype(np.float64), si, bs, float(mc), float(es))].astype(np.float32)

@njit(cache=True)
def _pf_ancc_seeded(seed, md_v, z_v, gr_v, gg, vmin, step, gs, ls, ir, N, ALPHA, RN, PN, IS, RP, RR, RESAMP):
    np.random.seed(seed)
    return _pf_ancc(md_v, z_v, gr_v, gg, vmin, step, gs, ls, ir, N, ALPHA, RN, PN, IS, RP, RR, RESAMP)


@njit(cache=True)
def _pf_z_seeded(seed, md_v, z_v, gr_v, gr_sm_v, gg_p, gg_s, vmin, step, gs, ip, iv, beta, icpt, zsig, N,
                 MOM, VN, PN, GR_WT, RP, RV, RESAMP):
    np.random.seed(seed)
    return _pf_z(md_v, z_v, gr_v, gr_sm_v, gg_p, gg_s, vmin, step, gs, ip, iv, beta, icpt, zsig, N,
                 MOM, VN, PN, GR_WT, RP, RV, RESAMP)


def run_pf_ancc(hw, tw_tvt, tw_gr, N=ANCC_N, seed=42):
    gs = _gr_sig(hw, tw_tvt, tw_gr); kn = hw[hw.TVT_input.notna()]; ev = hw[hw.TVT_input.isna()]
    if len(ev) == 0: return np.array([]), np.array([])
    ls = float(kn.TVT_input.iloc[-1]+kn.Z.iloc[-1])
    tail = kn.tail(30); dt = np.diff(tail.TVT_input.values); dz = np.diff(tail.Z.values); dm = np.diff(tail.MD.values); m = dm > 0
    ir = float(np.median((dt+dz)[m]/dm[m])) if m.sum() >= 3 else 0.
    gg, gmin, gst = _grid(tw_tvt, tw_gr)
    pts, std = _pf_ancc_seeded(
        int(seed),
        ev.MD.values.astype(np.float64),
        ev.Z.values.astype(np.float64),
        ev.GR.values.astype(np.float64),
        gg,
        gmin,
        gst,
        gs,
        ls,
        ir,
        N,
        ANCC_ALPHA,
        ANCC_RN,
        ANCC_PN,
        ANCC_IS,
        ANCC_RP,
        ANCC_RR,
        PF_RESAMP,
    )
    return pts.astype(np.float32), std.astype(np.float32)

def run_pf_z(hw, tw_tvt, tw_gr, N=PF_N, seed=42):
    gs = _gr_sig(hw, tw_tvt, tw_gr); tw_s = pd.Series(tw_gr).rolling(PF_GR_WIN, center=True, min_periods=1).mean().values.astype(np.float32)
    kna = hw[hw.TVT_input.notna()]; ev = hw[hw.TVT_input.isna()]
    if len(ev) == 0: return np.array([]), np.array([])
    dz_k = np.diff(kna.Z.values); dvt = np.diff(kna.TVT_input.values); dmd_k = np.diff(kna.MD.values); m2 = dmd_k > 0
    if m2.sum() >= 10:
        vz = dz_k[m2]/dmd_k[m2]; vt = dvt[m2]/dmd_k[m2]; A = np.column_stack([vz, np.ones_like(vz)])
        c, _, _, _ = np.linalg.lstsq(A, vt, rcond=None)
        beta, icpt, zsig = float(c[0]), float(c[1]), max(float(np.std(vt-(c[0]*vz+c[1]))), 0.001)
    else: beta, icpt, zsig = -1., 0., 0.1
    t2 = kna.tail(20); dvt2 = np.diff(t2.TVT_input.values); dmd2 = np.diff(t2.MD.values); m3 = dmd2 > 0
    iv = float(np.median(dvt2[m3]/dmd2[m3])) if m3.sum() >= 3 else 0.
    gg, gmin, gst = _grid(tw_tvt, tw_gr); gs2, _, _ = _grid(tw_tvt, tw_s)
    gr_sm = hw.GR.rolling(PF_GR_WIN, center=True, min_periods=1).mean()
    pts, std = _pf_z_seeded(
        int(seed),
        ev.MD.values.astype(np.float64),
        ev.Z.values.astype(np.float64),
        ev.GR.values.astype(np.float64),
        gr_sm.loc[ev.index].values.astype(np.float64),
        gg,
        gs2,
        gmin,
        gst,
        gs,
        float(kna.TVT_input.iloc[-1]),
        iv,
        beta,
        icpt,
        zsig,
        N,
        PF_MOM,
        PF_VN,
        PF_PN,
        PF_GR_WT,
        PF_ROUGH_P,
        PF_ROUGH_V,
        PF_RESAMP,
    )
    return pts.astype(np.float32), std.astype(np.float32)

def multi_scale_ncc(kgr, ktvt, hgr, hws=(8, 15, 25), stride=3):
    out = []
    for hw in hws:
        win = 2*hw+1; nk = len(kgr); nh = len(hgr)
        if nk < win+1 or nh == 0:
            out.append((np.full(nh, ktvt[-1], np.float32), np.zeros(nh, np.float32))); continue
        kg = pd.Series(kgr).rolling(5, center=True, min_periods=1).mean().values.astype(np.float32)
        hg = pd.Series(hgr).rolling(5, center=True, min_periods=1).mean().values.astype(np.float32)
        sts = np.arange(0, nk-win+1, stride, dtype=np.int32)
        if len(sts) == 0:
            out.append((np.full(nh, ktvt[-1], np.float32), np.zeros(nh, np.float32))); continue
        C = kg[sts[:, None]+np.arange(win, dtype=np.int32)[None, :]].astype(np.float32)
        Cn = (C-C.mean(1, keepdims=True))/(C.std(1, keepdims=True)+1e-6)
        hp = np.pad(hg, hw, mode="edge"); H = hp[np.arange(nh)[:, None]+np.arange(win)[None, :]].astype(np.float32)
        Hn = (H-H.mean(1, keepdims=True))/(H.std(1, keepdims=True)+1e-6)
        ncc = Hn@Cn.T/win; best = ncc.argmax(1); score = ncc.max(1).astype(np.float32)
        out.append((ktvt[np.clip(sts[best]+hw, 0, nk-1)].astype(np.float32), score))
    tvts = np.stack([o[0] for o in out], 1); scores = np.stack([o[1] for o in out], 1)
    sw = np.exp(3.*scores); sw /= sw.sum(1, keepdims=True)+1e-9
    return out, (tvts*sw).sum(1).astype(np.float32)

# %%
# ---- 128-seed likelihood-weighted particle filter (the workhorse), numba ---------
@njit(cache=True, nogil=True)
def _pf_lik_allseeds(md_v, z_v, gr_v, gg, vmin, step, gs, ls, ir, N, n_seeds, seed_base,
                     MOM, VN, PN, RP, RR, RESAMP, init_spr):
    n = len(md_v); preds = np.empty((n_seeds, n)); liks = np.empty(n_seeds); tmax = vmin + len(gg)*step
    for s in range(n_seeds):
        np.random.seed(seed_base + s)
        pos = np.empty(N); rate = np.empty(N); w = np.ones(N)/N
        for j in range(N):
            pos[j] = ls + init_spr*np.random.randn(); rate[j] = ir + 0.01*np.random.randn()
        log_lik = 0.0; prev_md = md_v[0] - 1.0
        for i in range(n):
            dm = md_v[i] - prev_md
            if dm < 1.0: dm = 1.0
            for j in range(N):
                rate[j] = MOM*rate[j] + VN*np.random.randn(); pos[j] += rate[j]*dm + PN*np.random.randn()
                tvt_j = pos[j] - z_v[i]
                if tvt_j < vmin-100.: tvt_j = vmin-100.
                if tvt_j > tmax+100.: tvt_j = tmax+100.
                pos[j] = tvt_j + z_v[i]
            avg_lk = 0.0
            for j in range(N):
                eg = _interp1(gg, pos[j]-z_v[i], vmin, step); d = (gr_v[i]-eg)/gs; dd = d*d
                if dd > 600.: dd = 600.
                lk = np.exp(-0.5*dd)
                if lk < 1e-300: lk = 1e-300
                avg_lk += w[j]*lk; w[j] = w[j]*lk
            if avg_lk < 1e-300: avg_lk = 1e-300
            log_lik += np.log(avg_lk)
            ws = 0.0
            for j in range(N): ws += w[j]
            if ws > 0.0:
                for j in range(N): w[j] /= ws
            else:
                for j in range(N): w[j] = 1./N
            neff = 0.0
            for j in range(N): neff += w[j]*w[j]
            neff = 1.0/neff
            if neff < RESAMP*N:
                cum = np.empty(N); c = 0.0
                for j in range(N): c += w[j]; cum[j] = c
                u0 = np.random.uniform(0., 1./N); newpos = np.empty(N); newrate = np.empty(N); ci = 0
                for j in range(N):
                    u = u0 + j/N
                    while ci < N-1 and cum[ci] < u: ci += 1
                    newpos[j] = pos[ci] + RP*np.random.randn(); newrate[j] = rate[ci] + RR*np.random.randn()
                for j in range(N): pos[j] = newpos[j]; rate[j] = newrate[j]; w[j] = 1./N
            est = 0.0
            for j in range(N): est += w[j]*(pos[j]-z_v[i])
            preds[s, i] = est; prev_md = md_v[i]
        liks[s] = log_lik
    return preds, liks

def lik_pf(hw, tw, n_particles=CFG.PF_PARTICLES, n_seeds=CFG.PF_SEEDS, scales=CFG.PF_SCALES,
           init_spr=4.5, seed_base=0, with_quality=False):
    """Likelihood-weighted PF ensemble. Returns ({pf_scale_X: pred_eval}, ev_index[, quality])."""
    tw_s = tw.sort_values("TVT"); tw_tvt = tw_s.TVT.values.astype(float)
    tw_gr = tw_s.GR.fillna(tw_s.GR.mean()).values.astype(float)
    kn = hw[hw.TVT_input.notna()]; ev = hw[hw.TVT_input.isna()]
    if len(ev) == 0: return {}, np.array([]), {}
    last = kn.iloc[-1]; ls = float(last.TVT_input) + float(last.Z)
    tw_at_k = np.interp(kn.TVT_input.values, tw_tvt, tw_gr)
    gs = float(np.clip(np.nanstd(kn.GR.fillna(0).values - tw_at_k), 10., 60.))
    tail = kn.tail(30); dt = np.diff(tail.TVT_input.values); dz = np.diff(tail.Z.values); dm = np.diff(tail.MD.values); m = dm > 0
    ir = float(np.median((dt+dz)[m]/dm[m])) if m.sum() >= 3 else 0.0
    gg, gmin, gst = _grid(tw_tvt, tw_gr)
    gr_v = hw.GR.interpolate(limit_direction="both").fillna(tw_gr.mean()).values.astype(float)[ev.index]
    preds, liks = _pf_lik_allseeds(ev.MD.values.astype(float), ev.Z.values.astype(float), gr_v,
                                   gg, gmin, gst, gs, ls, ir, n_particles, n_seeds, seed_base,
                                   0.998, 0.002, 0.005, 0.1, 0.001, 0.5, init_spr)
    ln = liks - liks.max(); out = {}
    for sc in scales:
        wts = np.exp(ln/float(sc)); wts /= wts.sum(); out[f"pf_scale_{sc:g}"] = (wts[:, None]*preds).sum(0)
    out["pf_mean"] = preds.mean(0)
    q = {}
    if with_quality:
        q = {"pf_best_ll": float(liks.max())/len(ev), "pf_ll_spread": float(liks.std()),
             "pf_pt_std": preds.std(0).astype(np.float32), "pf_gr_sig": gs}
    return out, ev.index.values, q

# JIT warm-up so timings below are representative
_m = np.linspace(1, 50, 20); _z = np.zeros(20); _g = np.full(20, 50.); _gg = np.linspace(45, 55, 100)
_pf_ancc(_m, _z, _g, _gg, 45., .1, 20., 50., 0., 8, .998, .002, .005, .3, .1, .001, .5)
_pf_z(_m, _z, _g, _g, _gg, _gg, 45., .1, 20., 50., 0., -1., 0., .1, 8, .993, .005, .01, .3, .2, .003, .5)
_beam_jit(np.linspace(-1., 1., 30), np.linspace(-1., 1., 50), 25, 8, 15., 100.)
_pf_lik_allseeds(_m, _z, _g, _gg, 45., .1, 20., 50., 0., 64, 4, 0, .998, .002, .005, .1, .001, .5, 4.5)
print("public replay trackers compiled.")
PLANE_K = 10; DENSE_SPW = 60; DENSE_K = 20

def robust_slope(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float); m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2 or np.std(x[m]) < 1e-6: return 0.
    return float(np.polyfit(x[m], y[m], 1)[0])

def affine_cal(kgr, tw_at_k, min_pts=20):
    v = np.isfinite(kgr) & np.isfinite(tw_at_k)
    if v.sum() < min_pts or np.std(tw_at_k[v]) < 1e-6:
        return 1., float(np.nanmean(kgr)-np.nanmean(tw_at_k)) if v.any() else 0.
    a, b = np.polyfit(tw_at_k[v], kgr[v], 1); return float(a), float(b)

def seg_b_well(ktvt, kz, form_col):
    bv = ktvt+kz-form_col; n = len(bv); b_full = float(np.median(bv))
    b_late = float(np.median(bv[max(0, n-50):])) if n >= 5 else b_full
    t1, t2 = n//3, 2*n//3
    b_early = float(np.median(bv[:max(1, t1)])) if t1 > 0 else b_full
    b_mid = float(np.median(bv[t1:max(t1+1, t2)])) if t2 > t1 else b_full
    w = np.exp(0.02*np.arange(n)); w /= w.sum()
    return b_full, b_early, b_mid, b_late, float(np.dot(w, bv))

class FormationPlaneKNN:
    def __init__(self, well_ids, data_dir):
        rows = []
        for wid in well_ids:
            try: df = pd.read_csv(data_dir/f"{wid}__horizontal_well.csv", usecols=["X","Y"]+FORMATIONS).dropna()
            except: continue
            if len(df) == 0: continue
            row = {"wid": wid, "x": float(df.X.median()), "y": float(df.Y.median())}
            for c in FORMATIONS: row[f"{c}_m"] = float(df[c].median())
            rows.append(row)
        self.df = pd.DataFrame(rows); self.wmap = {w: i for i, w in enumerate(self.df.wid)}
        xy = self.df[["x","y"]].to_numpy(); self.scale = np.where(xy.std(0) < 1e-3, 1., xy.std(0))
        self.tree = cKDTree(xy/self.scale); self.xa = self.df.x.to_numpy(); self.ya = self.df.y.to_numpy()
        self.fa = self.df[[f"{c}_m" for c in FORMATIONS]].to_numpy(np.float64)
    def impute(self, xy_q, self_wid=None, k=PLANE_K):
        q = xy_q/self.scale; nf = min(k+5, len(self.df)); dist, idx = self.tree.query(q, k=nf, workers=-1)
        if self_wid in self.wmap: dist = np.where(idx == self.wmap[self_wid], np.inf, dist)
        ordr = np.argpartition(dist, min(k-1, nf-1), 1)[:, :k]
        dk = np.take_along_axis(dist, ordr, 1); ik = np.take_along_axis(idx, ordr, 1)
        vk = np.isfinite(dk); w = np.where(vk, 1./(dk+1e-3), 0.).astype(np.float64)
        xn = self.xa[ik]; yn = self.ya[ik]; fn = self.fa[ik]; wx = w*xn; wy = w*yn
        A = np.zeros((len(q), 3, 3))
        A[:,0,0]=(wx*xn).sum(1); A[:,0,1]=(wx*yn).sum(1); A[:,0,2]=wx.sum(1)
        A[:,1,0]=A[:,0,1]; A[:,1,1]=(wy*yn).sum(1); A[:,1,2]=wy.sum(1)
        A[:,2,0]=A[:,0,2]; A[:,2,1]=A[:,1,2]; A[:,2,2]=w.sum(1)
        A[:,0,0]+=1e-9; A[:,1,1]+=1e-9; A[:,2,2]+=1e-9
        rhs = np.stack([(wx[:,:,None]*fn).sum(1), (wy[:,:,None]*fn).sum(1), (w[:,:,None]*fn).sum(1)], 1)
        try: coef = np.linalg.solve(A, rhs)
        except:
            coef = np.zeros((len(q), 3, 6))
            for r in range(len(q)):
                try: coef[r] = np.linalg.pinv(A[r])@rhs[r]
                except: pass
        Xq = xy_q[:,0]; Yq = xy_q[:,1]
        pred = (Xq[:,None]*coef[:,0,:]+Yq[:,None]*coef[:,1,:]+coef[:,2,:]).astype(np.float32)
        pred[~vk.any(1)] = self.fa.mean(0)
        return pred, np.where(vk, dk, np.inf).min(1).astype(np.float32)

class DenseANCCImputer:
    def __init__(self, well_ids, data_dir, spw=DENSE_SPW):
        xs, ys, an, wd = [], [], [], []
        for wid in well_ids:
            try: df = pd.read_csv(data_dir/f"{wid}__horizontal_well.csv", usecols=["X","Y","ANCC"]).dropna()
            except: continue
            if len(df) == 0: continue
            ix = np.linspace(0, len(df)-1, min(spw, len(df)), dtype=int); s = df.iloc[ix]
            xs.append(s.X.values); ys.append(s.Y.values); an.append(s.ANCC.values); wd.extend([wid]*len(s))
        self.xy = np.column_stack([np.concatenate(xs), np.concatenate(ys)])
        self.ancc = np.concatenate(an).astype(np.float32); self.wids = np.array(wd)
        self.scale = np.where(self.xy.std(0) < 1e-3, 1., self.xy.std(0)); self.tree = cKDTree(self.xy/self.scale)
    def impute(self, xy_q, self_wid=None, k=DENSE_K, nfetch=5000):
        xy_q = np.atleast_2d(xy_q); q = xy_q/self.scale; nf = min(nfetch, len(self.ancc))
        dist, idx = self.tree.query(q, k=nf, workers=-1)
        if self_wid: dist = np.where(self.wids[idx] == self_wid, np.inf, dist)
        ordr = np.argpartition(dist, min(k-1, nf-1), 1)[:, :k]
        dk = np.take_along_axis(dist, ordr, 1); ik = np.take_along_axis(idx, ordr, 1)
        vk = np.isfinite(dk); w = np.where(vk, 1./(dk+1e-3), 0.); sw = w.sum(1); safe = np.where(sw < 1e-9, 1., sw)
        a = self.ancc[ik]; ap = (a*w).sum(1)/safe; ap = np.where(sw < 1e-9, float(self.ancc.mean()), ap)
        var = ((a-ap[:,None])**2*w).sum(1)/safe
        return ap.astype(np.float32), np.sqrt(np.maximum(var, 0.)).astype(np.float32), np.where(vk, dk, np.inf).min(1).astype(np.float32)

_FI = None; _DI = None
ANCH_OFFS = np.array([-80,-40,-20,-10,-5,0,5,10,20,40,80], np.float32)
BEAM_OFFS = np.array([-40,-20,-10,-5,-3,0,3,5,10,20,40], np.float32)
SC_OFFS = np.array([-30,-15,-8,-4,-2,0,2,4,8,15,30], np.float32)
PF_OFFS = SC_OFFS.copy()

# %%
def build_well(hw_path, tw_path, is_train, likpf_map=None):
    global _FI, _DI
    wid = Path(hw_path).stem.replace("__horizontal_well", "")
    try: hw = pd.read_csv(hw_path); tw = pd.read_csv(tw_path).sort_values("TVT")
    except: return None
    if is_train and "TVT" not in hw.columns: return None
    kn = hw[hw.TVT_input.notna()]; ev = hw[hw.TVT_input.isna()]
    if len(ev) == 0 or len(kn) < 10: return None
    if is_train and hw.TVT.isna().all(): return None
    tw_tvt = tw.TVT.to_numpy(np.float32); tw_gr = tw.GR.to_numpy(np.float32)
    if len(tw_tvt) < 3: return None
    pf_a, std_a = run_pf_ancc(hw, tw_tvt, tw_gr, seed=stable_seed("pf_ancc", wid))
    if len(pf_a) == 0: return None
    pf_z, std_z = run_pf_z(hw, tw_tvt, tw_gr, seed=stable_seed("pf_z", wid))
    pf_use = pf_a.astype(np.float32); std_use = std_a.astype(np.float32)
    has_z = len(pf_z) == len(pf_a) and not np.any(np.isnan(pf_z))
    lk = kn.iloc[-1]; last_tvt = float(lk.TVT_input)
    gr_full = hw.GR.astype(float).interpolate(limit_direction="both").fillna(float(np.nanmean(tw_gr)))
    hgr = gr_full.iloc[ev.index[0]:].to_numpy(np.float32); kgr = gr_full.iloc[:len(kn)].to_numpy(np.float32)
    bpaths = {tag: beam_search(hgr, tw_tvt, tw_gr, last_tvt, bs, mc, es, r) for (bs, mc, es, r, tag) in BEAMS}
    beam_ref = (bpaths["cons"]+bpaths["sm5"])/2.
    ktvt = kn.TVT_input.to_numpy(np.float32)
    sc_res, sc_ens = multi_scale_ncc(kgr, ktvt, hgr, hws=(8, 15, 25), stride=3)
    sc8, sc8s = sc_res[0]; sc15, sc15s = sc_res[1]; sc25, sc25s = sc_res[2]; sc_cons = (sc8+sc15+sc25)/3.
    sc_trust = float(np.clip(len(kn)/200., 0., 0.6)); hyb_ref = (1-sc_trust)*beam_ref+sc_trust*sc_ens
    tw_at_k = np.interp(ktvt, tw_tvt, tw_gr).astype(np.float32); a_cal, b_cal = affine_cal(kgr, tw_at_k)
    kmd = kn.MD.to_numpy(np.float32); kz = kn.Z.to_numpy(np.float32)
    pfx_rmse = float(np.sqrt(np.mean((kgr-tw_at_k)**2)))
    slp_all = robust_slope(kmd, ktvt); slp_50 = robust_slope(kmd[-50:], ktvt[-50:]); slp_z = robust_slope(kz, ktvt)
    swid = wid if is_train else None
    xy_ev = ev[["X","Y"]].to_numpy(np.float64); xy_kn = kn[["X","Y"]].to_numpy(np.float64)
    form_ev, knn_d = _FI.impute(xy_ev, self_wid=swid); form_kn, _ = _FI.impute(xy_kn, self_wid=swid)
    z_kn = kn.Z.to_numpy(np.float32); z_ev = ev.Z.to_numpy(np.float32)
    tvt_fs = {}; form_rmse = {}; form_list = []
    for fi2, fn in enumerate(FORMATIONS):
        b_full, b_early, b_mid, b_late, b_wls = seg_b_well(ktvt, z_kn, form_kn[:, fi2])
        tvt_f = (-z_ev+form_ev[:, fi2]+b_full).astype(np.float32)
        tvt_fs[f"tvtF_{fn}"]=tvt_f; tvt_fs[f"tvtFw_{fn}"]=(-z_ev+form_ev[:,fi2]+b_wls).astype(np.float32)
        tvt_fs[f"tvtF50_{fn}"]=(-z_ev+form_ev[:,fi2]+b_late).astype(np.float32)
        tvt_fs[f"bw_{fn}"]=np.float32(b_full); tvt_fs[f"bww_{fn}"]=np.float32(b_wls); tvt_fs[f"bw50_{fn}"]=np.float32(b_late)
        tvt_fs[f"bw_early_{fn}"]=np.float32(b_early); tvt_fs[f"bw_mid_{fn}"]=np.float32(b_mid)
        form_rmse[fn]=float(np.sqrt(np.mean((ktvt-(-z_kn+form_kn[:,fi2]+b_full))**2))); form_list.append(tvt_f)
    fs = np.stack(form_list, 1)
    form_mean_d=(fs.mean(1)-last_tvt).astype(np.float32); form_std_d=fs.std(1).astype(np.float32); form_rng_d=(fs.max(1)-fs.min(1)).astype(np.float32)
    d_ancc, d_std, d_dist = _DI.impute(xy_ev, self_wid=swid); d_kn, d_std_kn, _ = _DI.impute(xy_kn, self_wid=swid)
    _, b_de, b_dm, b_dl, b_dw = seg_b_well(ktvt, z_kn, d_kn); b_d = float(np.median(ktvt+z_kn-d_kn))
    tvt_dense=(-z_ev+d_ancc+b_d).astype(np.float32); tvt_densew=(-z_ev+d_ancc+b_dw).astype(np.float32); tvt_dense50=(-z_ev+d_ancc+b_dl).astype(np.float32)
    res_kn = ktvt+z_kn-d_kn; d_rmse=float(np.sqrt(np.mean(res_kn**2))); d_bias=float(np.mean(res_kn)); d_nb_std=float(np.mean(d_std_kn))
    all_sigs=[pf_use]+list(bpaths.values())+[sc8,sc15,sc25,sc_ens,tvt_fs["tvtF_ANCC"],tvt_dense]
    sig_mat=np.stack(all_sigs,1); sig_std=sig_mat.std(1).astype(np.float32); sig_mean=(sig_mat.mean(1)-last_tvt).astype(np.float32)
    gr_s=pd.Series(gr_full.values); rolls={}
    for w in [5,21,51,101]:
        r=gr_s.rolling(w,center=True,min_periods=1); rolls[f"grm{w}"]=r.mean().iloc[ev.index].values.astype(np.float32); rolls[f"grs{w}"]=r.std().fillna(0).iloc[ev.index].values.astype(np.float32)
    for lag in [1,5,15,30]:
        rolls[f"glag{lag}"]=gr_s.shift(lag).bfill().iloc[ev.index].values.astype(np.float32); rolls[f"glead{lag}"]=gr_s.shift(-lag).ffill().iloc[ev.index].values.astype(np.float32)
    gr_d1=gr_s.diff().fillna(0.).iloc[ev.index].values.astype(np.float32); gr_d2=gr_s.diff().diff().fillna(0.).iloc[ev.index].values.astype(np.float32)
    gr_env=gr_s.rolling(21,center=True,min_periods=1).max().iloc[ev.index].values.astype(np.float32)
    gr_nrg=np.sqrt(np.maximum((gr_s**2).rolling(21,center=True,min_periods=1).mean(),0.)).iloc[ev.index].values.astype(np.float32)
    hmd=ev.MD.to_numpy(np.float32); md_since=hmd-float(lk.MD)
    slp_b_all=(last_tvt+slp_all*md_since).astype(np.float32); slp_b_50=(last_tvt+slp_50*md_since).astype(np.float32)
    mdd=hw.MD.diff().replace(0,np.nan)
    dzdmd=(hw.Z.diff()/mdd).iloc[ev.index].values.astype(np.float32); dxdmd=(hw.X.diff()/mdd).iloc[ev.index].values.astype(np.float32); dydmd=(hw.Y.diff()/mdd).iloc[ev.index].values.astype(np.float32)
    nh=len(ev); frac=(np.arange(nh)/max(nh-1,1)).astype(np.float32)
    def sc(v): return np.full(nh, np.float32(v), np.float32)
    feats={"well":wid,"id":[f"{wid}_{i}" for i in ev.index],"last_known_tvt":sc(last_tvt),
        "pf_ancc":pf_use,"pf_ancc_std":std_use,"pf_ancc_delta":(pf_use-last_tvt).astype(np.float32),
        "pf_z":(pf_z.astype(np.float32) if has_z else sc(last_tvt)),"pf_z_delta":((pf_z-last_tvt).astype(np.float32) if has_z else sc(0.)),
        "pf_vs_z":((pf_use-pf_z.astype(np.float32)) if has_z else sc(0.)),
        **{f"beam_{t}_d":(p-np.float32(last_tvt)).astype(np.float32) for t,p in bpaths.items()},
        "beam_mean_d":np.stack([(p-last_tvt) for p in bpaths.values()],1).mean(1).astype(np.float32),
        "beam_std_d":np.stack([(p-last_tvt) for p in bpaths.values()],1).std(1).astype(np.float32),
        "beam_med_d":np.median(np.stack([(p-last_tvt) for p in bpaths.values()],1),1).astype(np.float32),
        "sc8_d":(sc8-np.float32(last_tvt)).astype(np.float32),"sc8_sc":sc8s,"sc15_d":(sc15-np.float32(last_tvt)).astype(np.float32),"sc15_sc":sc15s,
        "sc25_d":(sc25-np.float32(last_tvt)).astype(np.float32),"sc25_sc":sc25s,"sc_cons_d":(sc_cons-np.float32(last_tvt)).astype(np.float32),
        "sc_ens_d":(sc_ens-np.float32(last_tvt)).astype(np.float32),"sc_trust":sc(sc_trust),"hyb_d":(hyb_ref-np.float32(last_tvt)).astype(np.float32),
        "sig_std":sig_std,"sig_mean_d":sig_mean,**tvt_fs,**{f"frm_rmse_{fn}":sc(form_rmse[fn]) for fn in FORMATIONS},
        "form_mean_d":form_mean_d,"form_std_d":form_std_d,"form_rng_d":form_rng_d,
        "spatial_ancc_d":(form_ev[:,0]-np.float32(np.interp(last_tvt,tw_tvt,tw_gr))),"spatial_knn_dist":knn_d,
        "dense_ancc":d_ancc,"dense_std":d_std,"dense_dist":d_dist,"tvt_dense_d":(tvt_dense-last_tvt).astype(np.float32),
        "tvt_densew_d":(tvt_densew-last_tvt).astype(np.float32),"tvt_dense50_d":(tvt_dense50-last_tvt).astype(np.float32),
        "dense_rmse":sc(d_rmse),"dense_bias":sc(d_bias),"dense_nb_std":sc(d_nb_std),
        "pf_vs_spatial":(pf_use-tvt_fs["tvtF_ANCC"]).astype(np.float32),"pf_vs_dense":(pf_use-tvt_dense).astype(np.float32),
        "spatial_vs_dense":(tvt_fs["tvtF_ANCC"]-tvt_dense).astype(np.float32),"beam_vs_spatial":(bpaths["cons"]-tvt_fs["tvtF_ANCC"]).astype(np.float32),
        "sc_vs_beam":(sc_ens-bpaths["cons"]).astype(np.float32),"cal_a":sc(a_cal),"cal_b":sc(b_cal),
        "pfx_rmse":sc(pfx_rmse),"known_len":sc(len(kn)),"eval_len":sc(nh),"slp_all":sc(slp_all),"slp_50":sc(slp_50),"slp_z":sc(slp_z),
        "slp_b_d_all":(slp_b_all-last_tvt).astype(np.float32),"slp_b_d_50":(slp_b_50-last_tvt).astype(np.float32),
        "ktvt_range":sc(float(np.ptp(ktvt))),"ktvt_std":sc(float(ktvt.std())),"md_since":md_since,"frac":frac,"frac2":frac**2,"sqrt_frac":np.sqrt(frac),
        "z":z_ev,"dx":(ev.X-float(lk.X)).to_numpy(np.float32),"dy":(ev.Y-float(lk.Y)).to_numpy(np.float32),"dz":(z_ev-float(lk.Z)).astype(np.float32),
        "dxy":np.sqrt((ev.X-float(lk.X))**2+(ev.Y-float(lk.Y))**2).to_numpy(np.float32),"dzdmd":dzdmd,"dxdmd":dxdmd,"dydmd":dydmd,
        "gr":hgr,"gr_d1":gr_d1,"gr_d2":gr_d2,"gr_env":gr_env,"gr_nrg":gr_nrg,
        "gr_vs_tw_anc":hgr-np.float32(np.interp(last_tvt,tw_tvt,tw_gr)),"gr_vs_slp_all":hgr-np.interp(slp_b_all,tw_tvt,tw_gr).astype(np.float32),
        **{f"tda{int(o)}":hgr-np.float32(np.interp(last_tvt+o,tw_tvt,tw_gr)) for o in ANCH_OFFS},
        **{f"tdbc{int(o)}":hgr-np.interp(beam_ref+o,tw_tvt,tw_gr).astype(np.float32) for o in BEAM_OFFS},
        **{f"tdsc{int(o)}":hgr-np.interp(sc_ens+o,tw_tvt,tw_gr).astype(np.float32) for o in SC_OFFS},
        **{f"tdpf{int(o)}":hgr-np.interp(pf_use+o,tw_tvt,tw_gr).astype(np.float32) for o in PF_OFFS},
        "tw_range":sc(float(np.ptp(tw_tvt))),"tw_gr_mean":sc(float(tw_gr.mean()))}
    for k,v in rolls.items(): feats[k]=v
    res = pd.DataFrame(feats)
    if is_train: res["target"]=(ev.TVT.to_numpy(np.float32)-np.float32(last_tvt))
    return res

def init_imputers(train_wids):
    global _FI, _DI
    _FI = FormationPlaneKNN(train_wids, CFG.DATA/"train"); _DI = DenseANCCImputer(train_wids, CFG.DATA/"train")

def _likpf_rows(wid, split):
    hw, tw = load_well(wid, split)
    out, idx, _ = lik_pf(hw, tw, seed_base=stable_seed("likpf", split, wid))
    if not len(out): return None
    d = {"id": [f"{wid}_{i}" for i in idx]}
    for k, v in out.items():
        d["likpf_" + k.replace("pf_scale_", "scale_").replace("pf_mean", "mean")] = v.astype(np.float32)
    return pd.DataFrame(d)

def build_likpf(wids, split):
    # threads are safe here: the lik-PF numba kernel is compiled with nogil=True, so it
    # releases the GIL and parallelises across threads (no pickling of numba code needed).
    res = Parallel(n_jobs=CFG.n_jobs, prefer="threads")(delayed(_likpf_rows)(w, split) for w in wids)
    return pd.concat([r for r in res if r is not None], ignore_index=True)

def build_features(wids, split, is_train):
    paths = [CFG.DATA/split/f"{w}__horizontal_well.csv" for w in wids]
    res = Parallel(n_jobs=CFG.n_jobs, prefer="threads")(
        delayed(build_well)(str(p), str(p.parent/f"{p.stem.replace('__horizontal_well','')}__typewell.csv"), is_train)
        for p in paths if (p.parent/f"{p.stem.replace('__horizontal_well','')}__typewell.csv").exists())
    parts = [r for r in res if r is not None]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

def add_likpf_features(df, likpf):
    df = df.merge(likpf, on="id", how="left")
    for c in [c for c in likpf.columns if c != "id"]:
        df[c] = df[c].fillna(df["last_known_tvt"]); df[c+"_d"] = (df[c]-df["last_known_tvt"]).astype(np.float32)
    return df

# %%
def _device():
    if CFG.USE_GPU == "cpu": return "cpu", "CPU"
    if CFG.USE_GPU == "gpu": return "gpu", "GPU"
    try:  # detect a real NVIDIA GPU (Kaggle GPU accelerator) via nvidia-smi
        import subprocess
        if subprocess.run(["nvidia-smi"], capture_output=True).returncode == 0:
            return "gpu", "GPU"
    except Exception:
        pass
    return "cpu", "CPU"

def lgb_configs(dev):
    base = dict(boosting_type="gbdt", objective="regression", verbose=-1, n_jobs=-1, max_bin=255)
    if dev == "gpu": base.update(device_type="gpu", gpu_use_dp=False)
    n = 600 if CFG.FAST else 5000
    return [
        dict(**base, num_leaves=255, min_child_samples=15, subsample=0.8, subsample_freq=1,
             colsample_bytree=0.8, reg_lambda=3.0, reg_alpha=0.05, learning_rate=0.03, n_estimators=n, seed=123),
        dict(**base, num_leaves=64, min_child_samples=40, subsample=0.474, subsample_freq=1,
             colsample_bytree=0.393, reg_lambda=95.75, reg_alpha=10.79, min_child_weight=0.24,
             learning_rate=0.0093, n_estimators=min(2*n, 10000), random_state=0),
        dict(**base, num_leaves=64, min_child_samples=40, subsample=0.474, subsample_freq=1,
             colsample_bytree=0.393, reg_lambda=95.75, reg_alpha=10.79, min_child_weight=0.24,
             learning_rate=0.0093, n_estimators=min(2*n, 10000), random_state=29),
    ]

def cb_configs(dev):
    tt = "GPU" if dev == "gpu" else "CPU"
    n = 800 if CFG.FAST else 8000
    return [
        dict(iterations=n, depth=7, l2_leaf_reg=2.0, min_data_in_leaf=15, border_count=254,
             loss_function="RMSE", task_type=tt, od_type="Iter", od_wait=300, verbose=0, learning_rate=0.02, random_seed=7),
        dict(iterations=n, depth=7, l2_leaf_reg=2.0, min_data_in_leaf=15, border_count=254,
             loss_function="RMSE", task_type=tt, od_type="Iter", od_wait=300, verbose=0, learning_rate=0.03, random_seed=123),
    ]

def train_stack(train_df, test_df, features):
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation
    from catboost import CatBoostRegressor
    from sklearn.model_selection import GroupKFold
    from sklearn.linear_model import Ridge
    dev, devname = _device(); print("device:", devname)
    X = train_df[features].values.astype(np.float32); y = train_df["target"].values.astype(np.float32)
    g = train_df["well"].values; Xt = test_df[features].values.astype(np.float32)
    cv = GroupKFold(CFG.n_splits); oof_cols = {}; test_cols = {}
    def run(name, make, fit_kw, is_lgb):
        # LightGBM: slice to best_iteration_ via num_iteration. CatBoost: use_best_model
        # already trims to the best tree, and its predict() takes no num_iteration kwarg.
        oof = np.zeros(len(train_df)); tp = np.zeros(len(test_df))
        for tr, va in cv.split(X, y, groups=g):
            m = make(); m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], **fit_kw)
            if is_lgb:
                it = m.best_iteration_
                oof[va] = m.predict(X[va], num_iteration=it); tp += m.predict(Xt, num_iteration=it) / CFG.n_splits
            else:
                oof[va] = m.predict(X[va]); tp += m.predict(Xt) / CFG.n_splits
        oof_cols[name] = oof; test_cols[name] = tp
        print(f"  {name}: OOF RMSE={rmse(y, oof):.4f}", flush=True)
    for i, p in enumerate(lgb_configs(dev)):
        run(f"lgb{i}", lambda p=p: LGBMRegressor(**p),
            dict(eval_metric="rmse", callbacks=[early_stopping(250, verbose=False), log_evaluation(0)]), True)
    for i, p in enumerate(cb_configs(dev)):
        run(f"cb{i}", lambda p=p: CatBoostRegressor(**p),
            dict(early_stopping_rounds=250, use_best_model=True), False)
    OOF = pd.DataFrame(oof_cols); TEST = pd.DataFrame(test_cols)
    rid = Ridge(alpha=1.66, positive=True, fit_intercept=True); meta = np.zeros(len(train_df))
    for tr, va in cv.split(OOF.values, y, groups=g):
        rid.fit(OOF.values[tr], y[tr]); meta[va] = rid.predict(OOF.values[va])
    rid.fit(OOF.values, y); meta_test = rid.predict(TEST.values)
    print(f"  ridge-stack OOF RMSE={rmse(y, meta):.4f}")
    return meta, meta_test, OOF, TEST

# %%

PUBLIC_SOURCE_PROVENANCE = {
    "pixiux_dual_pipeline_blend": "docs/notebooks/rogii-wellbore-geology-prediction/score_ascending_20260611/pixiux__rogii-dual-pipeline-blend/rogii-dual-pipeline-blend.ipynb",
    "ravaghi_lightgbm": "docs/notebooks/rogii-wellbore-geology-prediction/vote_top_20260611/ravaghi__wellbore-geology-prediction-lightgbm/wellbore-geology-prediction-lightgbm.ipynb",
    "note": "The base feature builder is the public Ravaghi-style builder copied inside the Pixiux notebook; Pixiux adds public likelihood-PF replay features via build_likpf/add_likpf_features.",
}


def list_train_wells(max_wells: int | None = None) -> list[str]:
    wids = sorted(p.stem.replace("__horizontal_well", "") for p in (CFG.DATA / "train").glob("*__horizontal_well.csv"))
    if CFG.N_TRAIN_WELLS:
        wids = wids[: CFG.N_TRAIN_WELLS]
    if max_wells is not None:
        wids = wids[: int(max_wells)]
    return wids


def list_test_wells(max_wells: int | None = None) -> list[str]:
    wids = sorted(p.stem.replace("__horizontal_well", "") for p in (CFG.DATA / "test").glob("*__horizontal_well.csv"))
    if max_wells is not None:
        wids = wids[: int(max_wells)]
    return wids


def build_replay_train_frames(max_wells: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    t0 = time.time()
    train_wids = list_train_wells(max_wells=max_wells)
    print(f"public replay train wells: {len(train_wids)} | n_jobs={CFG.n_jobs} | PF seeds={CFG.PF_SEEDS} | particles={CFG.PF_PARTICLES}", flush=True)
    init_imputers(train_wids)
    print("building Ravaghi/public base features from raw competition files...", flush=True)
    ravaghi_df = build_features(train_wids, "train", is_train=True).reset_index(drop=True)
    print(f"base features: rows={len(ravaghi_df):,} cols={len(ravaghi_df.columns)} elapsed={time.time()-t0:.1f}s", flush=True)
    print("building Pixiux public likelihood-PF replay features from raw competition files...", flush=True)
    likpf_train = build_likpf(train_wids, "train")
    pixiux_df = add_likpf_features(ravaghi_df.copy(), likpf_train).reset_index(drop=True)
    print(f"pixiux features: rows={len(pixiux_df):,} cols={len(pixiux_df.columns)} elapsed={time.time()-t0:.1f}s", flush=True)
    meta = {
        "train_wells": len(train_wids),
        "ravaghi_rows": int(len(ravaghi_df)),
        "pixiux_rows": int(len(pixiux_df)),
        "likpf_rows": int(len(likpf_train)),
        "elapsed_feature_seconds": round(time.time() - t0, 3),
        "provenance": PUBLIC_SOURCE_PROVENANCE,
    }
    return ravaghi_df, pixiux_df, meta


def build_replay_inference_frames(max_wells: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    t0 = time.time()
    train_wids = list_train_wells(max_wells=max_wells)
    test_wids = list_test_wells()
    print(
        f"public replay inference wells: train={len(train_wids)} test={len(test_wids)} "
        f"| n_jobs={CFG.n_jobs} | PF seeds={CFG.PF_SEEDS} | particles={CFG.PF_PARTICLES}",
        flush=True,
    )
    init_imputers(train_wids)
    print("building Pixiux/public base features from raw train files...", flush=True)
    train_base = build_features(train_wids, "train", is_train=True).reset_index(drop=True)
    print(f"train base features: rows={len(train_base):,} cols={len(train_base.columns)} elapsed={time.time()-t0:.1f}s", flush=True)
    print("building Pixiux/public base features from raw test files...", flush=True)
    test_base = build_features(test_wids, "test", is_train=False).reset_index(drop=True)
    print(f"test base features: rows={len(test_base):,} cols={len(test_base.columns)} elapsed={time.time()-t0:.1f}s", flush=True)
    print("building Pixiux likelihood-PF replay features for train...", flush=True)
    likpf_train = build_likpf(train_wids, "train")
    print("building Pixiux likelihood-PF replay features for test...", flush=True)
    likpf_test = build_likpf(test_wids, "test")
    train_df = add_likpf_features(train_base, likpf_train).reset_index(drop=True)
    test_df = add_likpf_features(test_base, likpf_test).reset_index(drop=True)
    meta = {
        "train_wells": len(train_wids),
        "test_wells": len(test_wids),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_likpf_rows": int(len(likpf_train)),
        "test_likpf_rows": int(len(likpf_test)),
        "elapsed_feature_seconds": round(time.time() - t0, 3),
        "provenance": PUBLIC_SOURCE_PROVENANCE,
    }
    return train_df, test_df, meta


def build_replay_test_frame() -> tuple[pd.DataFrame, dict[str, Any]]:
    t0 = time.time()
    train_wids = list_train_wells()
    test_wids = list_test_wells()
    print(
        f"public replay test wells: train_imputer_wells={len(train_wids)} test={len(test_wids)} "
        f"| n_jobs={CFG.n_jobs} | PF seeds={CFG.PF_SEEDS} | particles={CFG.PF_PARTICLES}",
        flush=True,
    )
    init_imputers(train_wids)
    print("building Pixiux/public base features from raw test files...", flush=True)
    test_base = build_features(test_wids, "test", is_train=False).reset_index(drop=True)
    print(f"test base features: rows={len(test_base):,} cols={len(test_base.columns)} elapsed={time.time()-t0:.1f}s", flush=True)
    print("building Pixiux likelihood-PF replay features for test...", flush=True)
    likpf_test = build_likpf(test_wids, "test")
    test_df = add_likpf_features(test_base, likpf_test).reset_index(drop=True)
    meta = {
        "train_imputer_wells": len(train_wids),
        "test_wells": len(test_wids),
        "test_rows": int(len(test_df)),
        "test_likpf_rows": int(len(likpf_test)),
        "elapsed_feature_seconds": round(time.time() - t0, 3),
        "provenance": PUBLIC_SOURCE_PROVENANCE,
    }
    return test_df, meta


def feature_columns_for_variant(df: pd.DataFrame, variant: str) -> list[str]:
    excluded = {"well", "id", "target"}
    features = [c for c in df.columns if c not in excluded]
    if variant == "pixiux_likpf_public_replay":
        features = [c for c in features if not (c.startswith("likpf_scale_") or c == "likpf_mean")]
    return features


def tracker_feature_columns(df: pd.DataFrame) -> list[str]:
    meta = [c for c in ["id", "well", "target", "last_known_tvt"] if c in df.columns]
    prefixes = (
        "pf_",
        "beam_",
        "sc",
        "hyb_",
        "sig_",
        "likpf_",
        "tdpf",
        "tdbc",
    )
    exact = {"pf_vs_z", "pf_vs_spatial", "pf_vs_dense", "beam_vs_spatial", "sc_vs_beam"}
    cols = [
        c
        for c in df.columns
        if c not in meta and (c in exact or any(c.startswith(prefix) for prefix in prefixes))
    ]
    return meta + cols


def save_tracker_feature_frame(df: pd.DataFrame, output_path: Path) -> dict[str, Any]:
    cols = tracker_feature_columns(df)
    frame = df[cols].copy()
    for col in frame.columns:
        if col not in {"id", "well"}:
            frame[col] = pd.to_numeric(frame[col], errors="coerce").astype(np.float32)
    frame.to_csv(output_path, index=False, compression="gzip")
    return {
        "path": output_path.name,
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "column_names": cols,
    }


def _fit_lgb_public_cv(
    df: pd.DataFrame,
    features: list[str],
    variant: str,
    model_output_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation

    dev, devname = _device()
    configs = lgb_configs(dev)
    print(f"training {variant}: rows={len(df):,} features={len(features)} device={devname} lgb_configs={len(configs)}", flush=True)
    X = df[features].to_numpy(np.float32)
    y = df["target"].to_numpy(np.float32)
    groups = df["well"].to_numpy()
    base = df["last_known_tvt"].to_numpy(np.float32)
    true_tvt = base + y
    cv = GroupKFold(CFG.n_splits)
    oof_by_model: list[np.ndarray] = []
    metric_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    prediction_rows: list[pd.DataFrame] = []
    model_rows: list[dict[str, Any]] = []
    if model_output_dir is not None:
        model_output_dir.mkdir(parents=True, exist_ok=True)

    for model_index, params in enumerate(configs):
        oof = np.zeros(len(df), dtype=np.float32)
        for fold, (tr, va) in enumerate(cv.split(X, y, groups=groups)):
            model = LGBMRegressor(**params)
            model.fit(
                X[tr],
                y[tr],
                eval_set=[(X[va], y[va])],
                eval_metric="rmse",
                callbacks=[early_stopping(250, verbose=False), log_evaluation(0)],
            )
            best_iter = int(model.best_iteration_ or params.get("n_estimators", 0))
            pred = model.predict(X[va], num_iteration=best_iter).astype(np.float32)
            oof[va] = pred
            model_file = None
            if model_output_dir is not None:
                model_file = f"{variant}__lgb{model_index}__fold{fold}.txt"
                model.booster_.save_model(str(model_output_dir / model_file), num_iteration=best_iter)
                model_rows.append(
                    {
                        "variant": variant,
                        "model": f"lgb{model_index}",
                        "model_index": int(model_index),
                        "fold": int(fold),
                        "best_iteration": best_iter,
                        "file": model_file,
                    }
                )
            fold_rmse = rmse(true_tvt[va], base[va] + pred)
            metric_rows.append(
                {
                    "variant": variant,
                    "model": f"lgb{model_index}",
                    "fold": int(fold),
                    "rows": int(len(va)),
                    "features": int(len(features)),
                    "best_iteration": best_iter,
                    "rmse_tvt": fold_rmse,
                    "rmse_residual": rmse(y[va], pred),
                }
            )
            for col, imp in zip(features, model.feature_importances_, strict=False):
                importance_rows.append(
                    {
                        "variant": variant,
                        "model": f"lgb{model_index}",
                        "fold": int(fold),
                        "feature": col,
                        "importance": float(imp),
                    }
                )
            print(f"  {variant} lgb{model_index} fold{fold}: rmse={fold_rmse:.6f} best_iter={best_iter}", flush=True)
        oof_by_model.append(oof)
        model_rmse = rmse(true_tvt, base + oof)
        metric_rows.append(
            {
                "variant": variant,
                "model": f"lgb{model_index}",
                "fold": "pooled",
                "rows": int(len(df)),
                "features": int(len(features)),
                "best_iteration": None,
                "rmse_tvt": model_rmse,
                "rmse_residual": rmse(y, oof),
            }
        )
        prediction_rows.append(
            pd.DataFrame(
                {
                    "id": df["id"].to_numpy(),
                    "well": groups,
                    "variant": variant,
                    "model": f"lgb{model_index}",
                    "target_tvt": true_tvt,
                    "last_known_tvt": base,
                    "pred_delta": oof,
                    "pred_tvt": base + oof,
                }
            )
        )
        print(f"  {variant} lgb{model_index}: pooled_rmse={model_rmse:.6f}", flush=True)

    ensemble = np.mean(np.vstack(oof_by_model), axis=0).astype(np.float32)
    metric_rows.append(
        {
            "variant": variant,
            "model": "lgb_mean",
            "fold": "pooled",
            "rows": int(len(df)),
            "features": int(len(features)),
            "best_iteration": None,
            "rmse_tvt": rmse(true_tvt, base + ensemble),
            "rmse_residual": rmse(y, ensemble),
        }
    )
    prediction_rows.append(
        pd.DataFrame(
            {
                "id": df["id"].to_numpy(),
                "well": groups,
                "variant": variant,
                "model": "lgb_mean",
                "target_tvt": true_tvt,
                "last_known_tvt": base,
                "pred_delta": ensemble,
                "pred_tvt": base + ensemble,
            }
        )
    )
    if model_output_dir is not None:
        manifest = {
            "experiment": "exp063_ravaghi_vs_pixiux_lgbm_feature_parity_audit",
            "mode": "strict_public_replay_saved_lgb_boosters",
            "variant": variant,
            "features": features,
            "n_features": int(len(features)),
            "n_splits": int(CFG.n_splits),
            "models": model_rows,
            "model_count": int(len(model_rows)),
            "base_prediction": "last_known_tvt",
            "target": "TVT - last_known_tvt",
            "lgb_configs": configs,
        }
        (model_output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"saved {len(model_rows)} LightGBM boosters to {model_output_dir}", flush=True)
    return pd.DataFrame(metric_rows), pd.DataFrame(importance_rows), pd.concat(prediction_rows, ignore_index=True)


def _fit_lgb_public_cv_predict(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list[str],
    variant: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation

    dev, devname = _device()
    configs = lgb_configs(dev)
    print(
        f"inference training {variant}: train_rows={len(train_df):,} test_rows={len(test_df):,} "
        f"features={len(features)} device={devname} lgb_configs={len(configs)}",
        flush=True,
    )
    for col in features:
        if col not in test_df.columns:
            test_df[col] = 0.0
    X = train_df[features].to_numpy(np.float32)
    Xt = test_df[features].to_numpy(np.float32)
    y = train_df["target"].to_numpy(np.float32)
    groups = train_df["well"].to_numpy()
    base = train_df["last_known_tvt"].to_numpy(np.float32)
    true_tvt = base + y
    test_base = test_df["last_known_tvt"].to_numpy(np.float32)
    cv = GroupKFold(CFG.n_splits)
    oof_by_model: list[np.ndarray] = []
    test_by_model: list[np.ndarray] = []
    metric_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    oof_prediction_rows: list[pd.DataFrame] = []
    test_prediction_rows: list[pd.DataFrame] = []

    for model_index, params in enumerate(configs):
        oof = np.zeros(len(train_df), dtype=np.float32)
        test_delta = np.zeros(len(test_df), dtype=np.float32)
        for fold, (tr, va) in enumerate(cv.split(X, y, groups=groups)):
            model = LGBMRegressor(**params)
            model.fit(
                X[tr],
                y[tr],
                eval_set=[(X[va], y[va])],
                eval_metric="rmse",
                callbacks=[early_stopping(250, verbose=False), log_evaluation(0)],
            )
            best_iter = int(model.best_iteration_ or params.get("n_estimators", 0))
            pred = model.predict(X[va], num_iteration=best_iter).astype(np.float32)
            test_fold = model.predict(Xt, num_iteration=best_iter).astype(np.float32)
            oof[va] = pred
            test_delta += test_fold / CFG.n_splits
            fold_rmse = rmse(true_tvt[va], base[va] + pred)
            metric_rows.append(
                {
                    "variant": variant,
                    "model": f"lgb{model_index}",
                    "fold": int(fold),
                    "rows": int(len(va)),
                    "features": int(len(features)),
                    "best_iteration": best_iter,
                    "rmse_tvt": fold_rmse,
                    "rmse_residual": rmse(y[va], pred),
                }
            )
            for col, imp in zip(features, model.feature_importances_, strict=False):
                importance_rows.append(
                    {
                        "variant": variant,
                        "model": f"lgb{model_index}",
                        "fold": int(fold),
                        "feature": col,
                        "importance": float(imp),
                    }
                )
            print(f"  {variant} lgb{model_index} fold{fold}: rmse={fold_rmse:.6f} best_iter={best_iter}", flush=True)
        oof_by_model.append(oof)
        test_by_model.append(test_delta)
        model_rmse = rmse(true_tvt, base + oof)
        metric_rows.append(
            {
                "variant": variant,
                "model": f"lgb{model_index}",
                "fold": "pooled",
                "rows": int(len(train_df)),
                "features": int(len(features)),
                "best_iteration": None,
                "rmse_tvt": model_rmse,
                "rmse_residual": rmse(y, oof),
            }
        )
        oof_prediction_rows.append(
            pd.DataFrame(
                {
                    "id": train_df["id"].to_numpy(),
                    "well": groups,
                    "variant": variant,
                    "model": f"lgb{model_index}",
                    "target_tvt": true_tvt,
                    "last_known_tvt": base,
                    "pred_delta": oof,
                    "pred_tvt": base + oof,
                }
            )
        )
        test_prediction_rows.append(
            pd.DataFrame(
                {
                    "id": test_df["id"].to_numpy(),
                    "well": test_df["well"].to_numpy(),
                    "variant": variant,
                    "model": f"lgb{model_index}",
                    "last_known_tvt": test_base,
                    "pred_delta": test_delta,
                    "pred_tvt": test_base + test_delta,
                }
            )
        )
        print(f"  {variant} lgb{model_index}: pooled_rmse={model_rmse:.6f}", flush=True)

    ensemble_oof = np.mean(np.vstack(oof_by_model), axis=0).astype(np.float32)
    ensemble_test = np.mean(np.vstack(test_by_model), axis=0).astype(np.float32)
    metric_rows.append(
        {
            "variant": variant,
            "model": "lgb_mean",
            "fold": "pooled",
            "rows": int(len(train_df)),
            "features": int(len(features)),
            "best_iteration": None,
            "rmse_tvt": rmse(true_tvt, base + ensemble_oof),
            "rmse_residual": rmse(y, ensemble_oof),
        }
    )
    oof_prediction_rows.append(
        pd.DataFrame(
            {
                "id": train_df["id"].to_numpy(),
                "well": groups,
                "variant": variant,
                "model": "lgb_mean",
                "target_tvt": true_tvt,
                "last_known_tvt": base,
                "pred_delta": ensemble_oof,
                "pred_tvt": base + ensemble_oof,
            }
        )
    )
    test_prediction_rows.append(
        pd.DataFrame(
            {
                "id": test_df["id"].to_numpy(),
                "well": test_df["well"].to_numpy(),
                "variant": variant,
                "model": "lgb_mean",
                "last_known_tvt": test_base,
                "pred_delta": ensemble_test,
                "pred_tvt": test_base + ensemble_test,
            }
        )
    )
    return (
        pd.DataFrame(metric_rows),
        pd.DataFrame(importance_rows),
        pd.concat(oof_prediction_rows, ignore_index=True),
        pd.concat(test_prediction_rows, ignore_index=True),
    )


def _sample_fallback_prediction(data_dir: Path, sample_ids: pd.Series, default_value: float) -> pd.Series:
    by_well: dict[str, float] = {}
    values: list[float] = []
    for raw_id in sample_ids.astype(str):
        try:
            wid, _row = raw_id.rsplit("_", 1)
        except ValueError:
            values.append(default_value)
            continue
        if wid not in by_well:
            path = data_dir / "test" / f"{wid}__horizontal_well.csv"
            if path.exists():
                hw = pd.read_csv(path, usecols=["TVT_input"])
                known = hw["TVT_input"].dropna()
                by_well[wid] = float(known.iloc[-1]) if len(known) else default_value
            else:
                by_well[wid] = default_value
        values.append(by_well[wid])
    return pd.Series(values, index=sample_ids.index, dtype="float64")


def _find_saved_lgb_model_dir(model_artifact_dir: str | Path | None = None) -> Path:
    candidates: list[Path] = []
    if model_artifact_dir is not None:
        candidates.append(Path(model_artifact_dir))
    candidates.extend(
        [
            CFG.OUT / "ravaghi_vs_pixiux_public_replay_lgb_models",
            Path.cwd() / "artifacts" / "ravaghi_vs_pixiux_public_replay_lgb_models",
            Path.cwd() / "ravaghi_vs_pixiux_public_replay_lgb_models",
        ]
    )
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        candidates.extend(kaggle_input.glob("**/ravaghi_vs_pixiux_public_replay_lgb_models"))
    for candidate in dict.fromkeys(candidates):
        if (candidate / "manifest.json").exists():
            return candidate
    checked = "\n".join(str(p) for p in candidates[:30])
    raise FileNotFoundError(f"No saved LightGBM model manifest found. Checked:\n{checked}")


def _predict_saved_lgb_ensemble(
    test_df: pd.DataFrame,
    model_dir: Path,
    *,
    variant: str,
    model_name: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    import lightgbm as lgb

    manifest = json.loads((model_dir / "manifest.json").read_text())
    if manifest.get("variant") != variant:
        raise ValueError(f"model manifest variant mismatch: {manifest.get('variant')} != {variant}")
    features = [str(c) for c in manifest["features"]]
    for col in features:
        if col not in test_df.columns:
            test_df[col] = 0.0
    Xt = test_df[features].to_numpy(np.float32)
    base = test_df["last_known_tvt"].to_numpy(np.float32)
    prediction_rows: list[pd.DataFrame] = []
    model_deltas: list[np.ndarray] = []
    models = manifest.get("models", [])
    for item in models:
        booster = lgb.Booster(model_file=str(model_dir / item["file"]))
        delta = booster.predict(Xt).astype(np.float32)
        model_deltas.append(delta)
        prediction_rows.append(
            pd.DataFrame(
                {
                    "id": test_df["id"].to_numpy(),
                    "well": test_df["well"].to_numpy(),
                    "variant": variant,
                    "model": item["model"],
                    "fold": int(item["fold"]),
                    "last_known_tvt": base,
                    "pred_delta": delta,
                    "pred_tvt": base + delta,
                }
            )
        )
    if not model_deltas:
        raise ValueError(f"No saved models listed in manifest: {model_dir / 'manifest.json'}")
    ensemble_delta = np.mean(np.vstack(model_deltas), axis=0).astype(np.float32)
    if model_name != "lgb_mean":
        model_items = [i for i, item in enumerate(models) if item["model"] == model_name]
        if not model_items:
            raise ValueError(f"No saved models for model_name={model_name}")
        ensemble_delta = np.mean(np.vstack([model_deltas[i] for i in model_items]), axis=0).astype(np.float32)
    prediction_rows.append(
        pd.DataFrame(
            {
                "id": test_df["id"].to_numpy(),
                "well": test_df["well"].to_numpy(),
                "variant": variant,
                "model": model_name,
                "fold": "mean",
                "last_known_tvt": base,
                "pred_delta": ensemble_delta,
                "pred_tvt": base + ensemble_delta,
            }
        )
    )
    meta = {
        "model_dir": str(model_dir),
        "model_count": int(len(models)),
        "feature_count": int(len(features)),
        "features": features,
        "manifest": manifest,
    }
    return pd.concat(prediction_rows, ignore_index=True), meta


def _plot_mean_importance(mean_importance: pd.DataFrame, output_path: Path, top_n: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    variants = mean_importance["variant"].drop_duplicates().tolist()
    if not variants:
        return
    fig_height = max(6.0, min(22.0, top_n * 0.32 + 2.0))
    fig, axes = plt.subplots(1, len(variants), figsize=(8.5 * len(variants), fig_height), squeeze=False)
    for ax, variant in zip(axes[0], variants, strict=False):
        subset = mean_importance[mean_importance["variant"] == variant].nlargest(top_n, "mean_importance")
        subset = subset.sort_values("mean_importance", ascending=True)
        ax.barh(subset["feature"], subset["mean_importance"], color="#2f6f8f")
        ax.set_title(f"{variant}\nmean over all public LGBM folds")
        ax.set_xlabel("mean feature_importances_")
        ax.grid(axis="x", alpha=0.25)
        ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def run_public_replay_audit(
    *,
    data_dir: str | Path,
    output_dir: str | Path,
    n_jobs: int | None = None,
    pf_seeds: int | None = None,
    pf_particles: int | None = None,
    fast: bool = False,
    use_gpu: str = "auto",
    max_wells: int | None = None,
    top_n_importance: int = 40,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_public_runtime(
        data_dir=data_dir,
        output_dir=output_dir,
        n_jobs=n_jobs,
        pf_seeds=pf_seeds,
        pf_particles=pf_particles,
        fast=fast,
        use_gpu=use_gpu,
        n_train_wells=max_wells,
    )
    t0 = time.time()
    ravaghi_df, pixiux_df, feature_meta = build_replay_train_frames(max_wells=max_wells)
    variants = {
        "ravaghi_public_lgbm_replay": ravaghi_df,
        "pixiux_likpf_public_replay": pixiux_df,
    }
    metrics_all: list[pd.DataFrame] = []
    importance_all: list[pd.DataFrame] = []
    predictions_all: list[pd.DataFrame] = []
    schema_rows: list[dict[str, Any]] = []
    model_dir = output_dir / "ravaghi_vs_pixiux_public_replay_lgb_models"
    for variant, frame in variants.items():
        features = feature_columns_for_variant(frame, variant)
        for col in features:
            schema_rows.append({"variant": variant, "feature": col})
        save_dir = model_dir if variant == "pixiux_likpf_public_replay" else None
        metrics, importance, predictions = _fit_lgb_public_cv(frame, features, variant, model_output_dir=save_dir)
        metrics_all.append(metrics)
        importance_all.append(importance)
        predictions_all.append(predictions)

    metrics_df = pd.concat(metrics_all, ignore_index=True)
    importance_df = pd.concat(importance_all, ignore_index=True)
    predictions_df = pd.concat(predictions_all, ignore_index=True)
    schema_df = pd.DataFrame(schema_rows)
    mean_importance = (
        importance_df.groupby(["variant", "feature"], as_index=False)
        .agg(
            mean_importance=("importance", "mean"),
            std_importance=("importance", "std"),
            fold_model_records=("importance", "size"),
        )
        .sort_values(["variant", "mean_importance"], ascending=[True, False])
    )
    metrics_df.to_csv(output_dir / "ravaghi_vs_pixiux_public_replay_metrics.csv", index=False)
    importance_df.to_csv(output_dir / "ravaghi_vs_pixiux_public_replay_feature_importance.csv", index=False)
    mean_importance.to_csv(output_dir / "ravaghi_vs_pixiux_public_replay_feature_importance_mean.csv", index=False)
    predictions_df.to_csv(output_dir / "ravaghi_vs_pixiux_public_replay_oof_predictions.csv.gz", index=False, compression="gzip")
    schema_df.to_csv(output_dir / "ravaghi_vs_pixiux_public_replay_feature_schema.csv", index=False)
    tracker_train_meta = save_tracker_feature_frame(
        pixiux_df,
        output_dir / "ravaghi_vs_pixiux_public_replay_tracker_features_train.csv.gz",
    )
    _plot_mean_importance(
        mean_importance,
        output_dir / "ravaghi_vs_pixiux_public_replay_feature_importance_mean_top.png",
        int(top_n_importance),
    )
    pooled = metrics_df[metrics_df["fold"].astype(str).eq("pooled")].copy()
    selected = pooled.sort_values("rmse_tvt").iloc[0].to_dict()
    summary = {
        "experiment": "exp063_ravaghi_vs_pixiux_lgbm_feature_parity_audit",
        "mode": "strict_public_notebook_raw_replay",
        "feature_meta": feature_meta,
        "selected": selected,
        "pooled_metrics": pooled.to_dict(orient="records"),
        "artifacts": {
            "metrics": "ravaghi_vs_pixiux_public_replay_metrics.csv",
            "feature_importance": "ravaghi_vs_pixiux_public_replay_feature_importance.csv",
            "mean_feature_importance": "ravaghi_vs_pixiux_public_replay_feature_importance_mean.csv",
            "mean_feature_importance_plot": "ravaghi_vs_pixiux_public_replay_feature_importance_mean_top.png",
            "oof_predictions": "ravaghi_vs_pixiux_public_replay_oof_predictions.csv.gz",
            "feature_schema": "ravaghi_vs_pixiux_public_replay_feature_schema.csv",
            "saved_lgb_models": "ravaghi_vs_pixiux_public_replay_lgb_models",
            "tracker_features_train": "ravaghi_vs_pixiux_public_replay_tracker_features_train.csv.gz",
        },
        "reusable_feature_frames": {
            "tracker_features_train": tracker_train_meta,
        },
        "elapsed_seconds": round(time.time() - t0, 3),
    }
    (output_dir / "ravaghi_vs_pixiux_public_replay_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def run_public_replay_inference(
    *,
    data_dir: str | Path,
    output_dir: str | Path,
    submission_path: str | Path,
    n_jobs: int | None = None,
    pf_seeds: int | None = None,
    pf_particles: int | None = None,
    fast: bool = False,
    use_gpu: str = "auto",
    max_wells: int | None = None,
    variant: str = "pixiux_likpf_public_replay",
    model_name: str = "lgb_mean",
    model_artifact_dir: str | Path | None = None,
    sample_submission_path: str | Path | None = None,
    submission_target_column: str = "tvt",
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    submission_path = Path(submission_path)
    configure_public_runtime(
        data_dir=data_dir,
        output_dir=output_dir,
        n_jobs=n_jobs,
        pf_seeds=pf_seeds,
        pf_particles=pf_particles,
        fast=fast,
        use_gpu=use_gpu,
        n_train_wells=max_wells,
    )
    t0 = time.time()
    if variant != "pixiux_likpf_public_replay":
        raise ValueError(f"Unsupported inference variant for exp063: {variant}")

    model_dir = _find_saved_lgb_model_dir(model_artifact_dir)
    print(f"loading saved LightGBM boosters from {model_dir}", flush=True)
    test_df, feature_meta = build_replay_test_frame()
    test_predictions_df, model_meta = _predict_saved_lgb_ensemble(
        test_df,
        model_dir,
        variant=variant,
        model_name=model_name,
    )
    features = model_meta["features"]
    schema_df = pd.DataFrame({"variant": variant, "feature": features})

    selected_test = test_predictions_df[
        (test_predictions_df["variant"].eq(variant)) & (test_predictions_df["model"].eq(model_name))
    ].copy()
    if selected_test.empty:
        raise ValueError(f"No test predictions for variant={variant} model={model_name}")
    sample_path = Path(sample_submission_path) if sample_submission_path is not None else CFG.DATA / "sample_submission.csv"
    sample = pd.read_csv(sample_path)
    if "id" not in sample.columns:
        raise ValueError(f"sample submission must contain id column: {sample_path}")
    if submission_target_column not in sample.columns:
        if len(sample.columns) >= 2:
            submission_target_column = str(sample.columns[1])
        else:
            raise ValueError(f"sample submission must contain a target column: {sample_path}")

    pred_map = dict(zip(selected_test["id"].astype(str), selected_test["pred_tvt"].astype(float), strict=False))
    default_value = float(selected_test["pred_tvt"].mean())
    fallback = _sample_fallback_prediction(CFG.DATA, sample["id"], default_value)
    mapped = sample["id"].astype(str).map(pred_map)
    missing_mask = mapped.isna()
    sample[submission_target_column] = mapped.fillna(fallback).astype("float64")
    sample.to_csv(submission_path, index=False)

    schema_df.to_csv(output_dir / "ravaghi_vs_pixiux_public_replay_inference_feature_schema.csv", index=False)
    test_predictions_df.to_csv(
        output_dir / "ravaghi_vs_pixiux_public_replay_inference_test_predictions.csv.gz",
        index=False,
        compression="gzip",
    )
    tracker_test_meta = save_tracker_feature_frame(
        test_df,
        output_dir / "ravaghi_vs_pixiux_public_replay_tracker_features_test.csv.gz",
    )
    diagnostics_df = pd.DataFrame(
        [
            {
                "variant": variant,
                "model": model_name,
                "test_rows": int(len(test_df)),
                "submission_rows": int(len(sample)),
                "predicted_rows": int((~missing_mask).sum()),
                "fallback_rows": int(missing_mask.sum()),
                "saved_model_count": int(model_meta["model_count"]),
                "feature_count": int(model_meta["feature_count"]),
                "model_dir": str(model_dir),
            }
        ]
    )
    diagnostics_df.to_csv(output_dir / "ravaghi_vs_pixiux_public_replay_inference_metrics.csv", index=False)
    diagnostics = {
        "rows": int(len(sample)),
        "predicted_rows": int((~missing_mask).sum()),
        "fallback_rows": int(missing_mask.sum()),
        "target_column": submission_target_column,
        "submission_path": str(submission_path),
        "prediction_mean": float(sample[submission_target_column].mean()),
        "prediction_std": float(sample[submission_target_column].std()),
        "prediction_min": float(sample[submission_target_column].min()),
        "prediction_max": float(sample[submission_target_column].max()),
    }
    summary = {
        "experiment": "exp063_ravaghi_vs_pixiux_lgbm_feature_parity_audit",
        "mode": "strict_public_replay_saved_model_inference_no_override",
        "variant": variant,
        "model": model_name,
        "feature_meta": feature_meta,
        "model_meta": {
            "model_dir": str(model_dir),
            "model_count": int(model_meta["model_count"]),
            "feature_count": int(model_meta["feature_count"]),
        },
        "reusable_feature_frames": {
            "tracker_features_test": tracker_test_meta,
        },
        "submission": diagnostics,
        "excluded": [
            "train feature regeneration",
            "inference-time LightGBM training",
            "hidden-specific branch",
            "guarded overlap override",
            "static visible override",
            "pretrained boosters",
            "CatBoost",
            "Ridge stack",
            "final public notebook blend",
            "projection postprocess",
        ],
        "artifacts": {
            "submission": str(submission_path.name),
            "metrics": "ravaghi_vs_pixiux_public_replay_inference_metrics.csv",
            "test_predictions": "ravaghi_vs_pixiux_public_replay_inference_test_predictions.csv.gz",
            "feature_schema": "ravaghi_vs_pixiux_public_replay_inference_feature_schema.csv",
            "tracker_features_test": "ravaghi_vs_pixiux_public_replay_tracker_features_test.csv.gz",
        },
        "elapsed_seconds": round(time.time() - t0, 3),
    }
    (output_dir / "ravaghi_vs_pixiux_public_replay_inference_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return summary
