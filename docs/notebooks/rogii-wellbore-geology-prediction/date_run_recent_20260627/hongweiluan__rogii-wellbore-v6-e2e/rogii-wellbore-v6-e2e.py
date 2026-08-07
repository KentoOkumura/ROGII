# %% [code]
import warnings; warnings.filterwarnings('ignore')
import sys, time, os
from pathlib import Path

# ═══════════════════════════════════════════════════════════════
# ALL PATHS — edit here only
# ═══════════════════════════════════════════════════════════════
_COMP   = Path('/kaggle/input/competitions/rogii-wellbore-geology-prediction')
TRAIN_DIR  = _COMP / 'train'
TEST_DIR   = _COMP / 'test'
COMP_DIR   = _COMP
OUT_DIR    = Path('/kaggle/working')

# GBM model weights + oof_meta (dataset: hongweiluan/model-v11)
MODELS_DS  = Path('/kaggle/input/datasets/hongweiluan/model-v11')
MODEL_DIR  = MODELS_DS          # Kaggle auto-unzips — all files at root

# TCN model weights (dataset: hongweiluan/rogii-tcn-v4-models2)
TCN_DS     = Path('/kaggle/input/datasets/hongweiluan/rogii-tcn-v4-models2')
# ═══════════════════════════════════════════════════════════════

import warnings as _w; _w.filterwarnings('ignore')
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy.spatial import cKDTree
from joblib import Parallel, delayed
from numba import njit
import pywt
import multiprocessing
import pandas as pd
import numpy as np

# v6: 5 LGBM + XGBoost + wavelet DWT features + fixed tau PP + formation TVT blend

NCPU    = min(8, multiprocessing.cpu_count())
N_SPLITS = 5
SEED    = 42

# ─────────────────────────── Constants ────────────────────────
FORMATIONS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]
PLANE_K    = 10
DENSE_K    = 20
DENSE_SPW  = 60

BEAMS = [
    (10, 20.0, 144.0, 2, 'cons'),
    (10,  8.0,  64.0, 2, 'loose'),
    ( 8, 35.0, 220.0, 1, 'vcons'),
    (10, 14.0,  90.0, 5, 'sm5'),
    (20,  4.0,  36.0, 3, 'vloose'),
    (12, 12.0, 100.0, 3, 'mid'),
    (15, 25.0, 180.0, 2, 'stiff'),
    # 7 additional configs from fle3n Engine A (wider search diversity)
    (20, 30.0, 200.0, 2, 'wide1'),
    (15, 10.0,  80.0, 4, 'wide2'),
    (25,  6.0,  50.0, 3, 'wide3'),
    (10, 40.0, 300.0, 1, 'stiff2'),
    (12, 18.0, 120.0, 5, 'mid2'),
    (30,  8.0,  70.0, 2, 'wide4'),
    (10, 50.0, 400.0, 0, 'vvstiff'),  # max penalty, no smoothing
]

PF_N          = 600  # fle3n uses N=600 for single-run PF
PF_MOM        = 0.993; PF_VN = 0.005; PF_PN = 0.01
PF_GR_SIG_MIN = 10.;   PF_GR_SIG_MAX = 60.; PF_GR_SIG_DEF = 30.
PF_INIT_SPR   = 0.5;   PF_RESAMP = 0.5
PF_ROUGH_P    = 0.2;   PF_ROUGH_V = 0.003
PF_GR_WIN     = 21;    PF_GR_WT  = 0.3
ANCC_ALPHA    = 0.998; ANCC_RN = 0.002; ANCC_PN = 0.005
ANCC_IS       = 0.3;   ANCC_RP = 0.1;  ANCC_RR = 0.001

DTW_RADII      = (20, 50, 100, 200)
DTW_STOCH_K    = 12
DTW_STOCH_TEMP = 3.0

ANCH_OFFS = np.array([-80,-40,-20,-10,-5, 0, 5,10,20,40,80], np.float32)
BEAM_OFFS = np.array([-40,-20,-10, -5,-3, 0, 3, 5,10,20,40], np.float32)
SC_OFFS   = np.array([-30,-15, -8, -4,-2, 0, 2, 4, 8,15,30], np.float32)
PF_OFFS   = np.array([-30,-15, -8, -4,-2, 0, 2, 4, 8,15,30], np.float32)
DTW_OFFS  = np.array([-20,-10, -5, -2, 0, 2, 5,10,20],        np.float32)

# ─────────────────────────── JIT kernels ──────────────────────
@njit(cache=True)
def _interp1(grid, v, vmin, step):
    i = int((v - vmin) / step)
    if i < 0: return grid[0]
    n = len(grid) - 1
    if i >= n: return grid[n]
    t = (v - vmin) / step - i
    return grid[i] * (1. - t) + grid[i + 1] * t

@njit(cache=True)
def _resamp(pos, aux, w, N, rp, rv):
    cum = np.zeros(N + 1)
    for j in range(N): cum[j+1] = cum[j] + w[j]
    u0 = np.random.uniform(0., 1./N)
    np2 = np.empty(N); na = np.empty(N); ci = 0
    for j in range(N):
        u = u0 + j / N
        while ci < N-1 and cum[ci+1] < u: ci += 1
        np2[j] = pos[ci] + rp * np.random.randn()
        na[j]  = aux[ci] + rv * np.random.randn()
    return np2, na

@njit(cache=True)
def _beam_jit(sgr, tw_gr, si, BS, mc, es):
    n = len(sgr); nt = len(tw_gr); MAX = BS * 6
    bidx = np.zeros(BS, np.int64); bidx[0] = si
    bcost = np.full(BS, 1e30); bcost[0] = 0.; bn = np.int64(1)
    hI = np.zeros((n, BS), np.int64); hP = np.zeros((n, BS), np.int64)
    cI = np.zeros(MAX, np.int64); cC = np.full(MAX, 1e30); cP = np.zeros(MAX, np.int64)
    for step in range(n):
        gv = sgr[step]; nc = np.int64(0)
        for bi in range(bn):
            idx = bidx[bi]; cost = bcost[bi]
            for d in range(-2, 3):
                ni = idx + d
                if ni < 0 or ni >= nt: continue
                tot = cost + (gv - tw_gr[ni])**2 / es + mc*(d if d >= 0 else -d)
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
                cI[i],cI[mi]=cI[mi],cI[i]; cC[i],cC[mi]=cC[mi],cC[i]; cP[i],cP[mi]=cP[mi],cP[i]
        hI[step,:kept]=cI[:kept]; hP[step,:kept]=cP[:kept]
        bidx[:kept]=cI[:kept]; bcost[:kept]=cC[:kept]; bn=kept
    best=np.int64(0)
    for b in range(1, bn):
        if bcost[b] < bcost[best]: best=b
    path=np.zeros(n, np.int64); b=best
    for s in range(n-1,-1,-1): path[s]=hI[s,b]; b=hP[s,b]
    return path

@njit(cache=True)
def _pf_ancc(md_v, z_v, gr_v, gg, vmin, step, gs, ls, ir, N,
             ALPHA, RN, PN, IS, RP, RR, RESAMP):
    pos=np.empty(N); rate=np.empty(N); w=np.ones(N)/N
    for j in range(N):
        pos[j]=ls+IS*np.random.randn(); rate[j]=ir+0.01*np.random.randn()
    pts=np.empty(len(md_v)); std_=np.empty(len(md_v)); pm=md_v[0]-1.
    for i in range(len(md_v)):
        dm=md_v[i]-pm; dm=max(dm,1.)
        for j in range(N):
            rate[j]=ALPHA*rate[j]+RN*np.random.randn()
            pos[j]+=rate[j]*dm+PN*np.random.randn()
            tvt_j=pos[j]-z_v[i]
            tvt_j=max(tvt_j,vmin-50.); tvt_j=min(tvt_j,vmin+len(gg)*step+50.)
            pos[j]=tvt_j+z_v[i]
        if not np.isnan(gr_v[i]):
            ws=0.
            for j in range(N):
                eg=_interp1(gg,pos[j]-z_v[i],vmin,step)
                d=(gr_v[i]-eg)/gs
                lk=max(np.exp(-0.5*d*d) if d*d<600. else 0.,1e-300)
                w[j]*=lk; ws+=w[j]
            if ws>0.:
                for j in range(N): w[j]/=ws
            else:
                for j in range(N): w[j]=1./N
        ne=0.
        for j in range(N): ne+=w[j]*w[j]
        if 1./ne<RESAMP*N:
            pos,rate=_resamp(pos,rate,w,N,RP,RR)
            for j in range(N): w[j]=1./N
        tv=0.
        for j in range(N): tv+=w[j]*(pos[j]-z_v[i])
        pts[i]=tv; va=0.
        for j in range(N): va+=w[j]*(pos[j]-z_v[i]-tv)**2
        std_[i]=va**0.5; pm=md_v[i]
    return pts, std_

@njit(cache=True)
def _pf_ancc_with_lik(md_v, z_v, gr_v, gg, vmin, step, gs, ls, ir, N,
                       ALPHA, RN, PN, IS, RP, RR, RESAMP, seed):
    """Like _pf_ancc but also returns total data log-likelihood for seed-ensemble weighting."""
    np.random.seed(seed)
    pos=np.empty(N); rate=np.empty(N); w=np.ones(N)/N
    for j in range(N):
        pos[j]=ls+IS*np.random.randn(); rate[j]=ir+0.01*np.random.randn()
    pts=np.empty(len(md_v)); pm=md_v[0]-1.; total_log_lik=0.0
    for i in range(len(md_v)):
        dm=md_v[i]-pm; dm=max(dm,1.)
        for j in range(N):
            rate[j]=ALPHA*rate[j]+RN*np.random.randn()
            pos[j]+=rate[j]*dm+PN*np.random.randn()
            tvt_j=pos[j]-z_v[i]
            tvt_j=max(tvt_j,vmin-50.); tvt_j=min(tvt_j,vmin+len(gg)*step+50.)
            pos[j]=tvt_j+z_v[i]
        if not np.isnan(gr_v[i]):
            ws=0.
            for j in range(N):
                eg=_interp1(gg,pos[j]-z_v[i],vmin,step)
                d=(gr_v[i]-eg)/gs
                lk=max(np.exp(-0.5*d*d) if d*d<600. else 0.,1e-300)
                w[j]*=lk; ws+=w[j]
            if ws>0.:
                total_log_lik+=np.log(ws/N+1e-300)
                for j in range(N): w[j]/=ws
            else:
                for j in range(N): w[j]=1./N
        ne=0.
        for j in range(N): ne+=w[j]*w[j]
        if 1./ne<RESAMP*N:
            pos,rate=_resamp(pos,rate,w,N,RP,RR)
            for j in range(N): w[j]=1./N
        tv=0.
        for j in range(N): tv+=w[j]*(pos[j]-z_v[i])
        pts[i]=tv; pm=md_v[i]
    return pts, total_log_lik

@njit(cache=True)
def _pf_z(md_v, z_v, gr_v, gr_sm_v, gg_p, gg_s, vmin, step,
          gs, ip, iv, beta, icpt, zsig, N,
          MOM, VN, PN, GR_WT, RP, RV, RESAMP):
    pos=np.empty(N); vel=np.empty(N); w=np.ones(N)/N
    for j in range(N):
        pos[j]=ip+0.5*np.random.randn(); vel[j]=iv+0.02*np.random.randn()
    pts=np.empty(len(md_v)); std_=np.empty(len(md_v)); pm=md_v[0]-1.; pz=z_v[0]-1.
    for i in range(len(md_v)):
        dm=md_v[i]-pm; dm=max(dm,1.)
        dzd=(z_v[i]-pz)/dm; ve=beta*dzd+icpt
        for j in range(N):
            vel[j]=MOM*vel[j]+VN*np.random.randn()
            pos[j]+=vel[j]*dm+PN*np.random.randn()
            pos[j]=max(pos[j],vmin-50.); pos[j]=min(pos[j],vmin+len(gg_p)*step+50.)
        if not np.isnan(gr_v[i]):
            ws=0.
            for j in range(N):
                ep=_interp1(gg_p,pos[j],vmin,step)
                dp=(gr_v[i]-ep)/gs
                lp=max(np.exp(-0.5*dp*dp) if dp*dp<600. else 0.,1e-300)
                if not np.isnan(gr_sm_v[i]):
                    es=_interp1(gg_s,pos[j],vmin,step)
                    ds=(gr_sm_v[i]-es)/(gs*1.5)
                    ls2=max(np.exp(-0.5*ds*ds) if ds*ds<600. else 0.,1e-300)
                    lk=(1.-GR_WT)*lp+GR_WT*ls2
                else:
                    lk=lp
                lk=max(lk,1e-300); w[j]*=lk; ws+=w[j]
            if ws>0.:
                for j in range(N): w[j]/=ws
            else:
                for j in range(N): w[j]=1./N
        ws2=0.
        for j in range(N):
            dv=(vel[j]-ve)/max(zsig*2.,0.005)
            lz=max(np.exp(-0.5*dv*dv) if dv*dv<600. else 0.,1e-300)
            w[j]*=lz; ws2+=w[j]
        if ws2>0.:
            for j in range(N): w[j]/=ws2
        else:
            for j in range(N): w[j]=1./N
        ne=0.
        for j in range(N): ne+=w[j]*w[j]
        if 1./ne<RESAMP*N:
            pos,vel=_resamp(pos,vel,w,N,RP,RV)
            for j in range(N): w[j]=1./N
        wm=0.
        for j in range(N): wm+=w[j]*pos[j]
        pts[i]=wm; va=0.
        for j in range(N): va+=w[j]*(pos[j]-wm)**2
        std_[i]=va**0.5; pm=md_v[i]; pz=z_v[i]
    return pts, std_

@njit(cache=True)
def _dtw_sakoe_chiba(query, ref, radius):
    N=len(query); M=len(ref); INF=1e18
    D=np.full((N,M),INF)
    slope=(M-1.0)/max(N-1.0,1.0)
    for i in range(N):
        j_center=int(round(i*slope))
        j_lo=max(0,j_center-radius); j_hi=min(M-1,j_center+radius)
        for j in range(j_lo,j_hi+1):
            cost=(query[i]-ref[j])**2
            if i==0 and j==0: D[i,j]=cost
            elif i==0:
                prev=D[i,j-1]; D[i,j]=cost+(prev if prev<INF else INF)
            elif j==0:
                prev=D[i-1,j]; D[i,j]=cost+(prev if prev<INF else INF)
            else:
                a=D[i-1,j-1]; b=D[i-1,j]; c=D[i,j-1]
                mn=a if a<b else b; mn=mn if mn<c else c
                D[i,j]=cost+(mn if mn<INF else INF)
    i=N-1; j=M-1
    pi=np.zeros(N+M,np.int64); pj=np.zeros(N+M,np.int64); k=0
    while i>0 or j>0:
        pi[k]=i; pj[k]=j; k+=1
        if i==0: j-=1
        elif j==0: i-=1
        else:
            a=D[i-1,j-1]; b=D[i-1,j]; c=D[i,j-1]
            if a<=b and a<=c: i-=1; j-=1
            elif b<=c: i-=1
            else: j-=1
    pi[k]=0; pj[k]=0; k+=1
    return D, pi[:k], pj[:k]

@njit(cache=True)
def _dtw_anchor_constrained(query, ref, radius, anchor_i, anchor_j, sigma=5.0):
    """Anchor-constrained DTW (US11391856 soft marker approach).

    At known anchor positions (anchor_i[k], anchor_j[k]), subtract a Gaussian
    log-likelihood from the cost matrix: cost[i,j] -= log N(j | anchor_j[k], sigma^2)
    This biases the warp path probabilistically through known TVT positions.

    Args:
        query: horizontal GR signal (N,)
        ref: typewell GR (M,)
        radius: Sakoe-Chiba band
        anchor_i: horizontal well row indices where TVT_input is known (K_anchors,)
        anchor_j: corresponding typewell indices (K_anchors,)
        sigma: constraint softness in typewell index units (~5 = ±5 typewell steps)
    """
    N=len(query); M=len(ref); INF=1e18
    D=np.full((N,M),INF)
    slope=(M-1.0)/max(N-1.0,1.0)
    inv2s2 = 1.0 / (2.0 * sigma * sigma)

    for i in range(N):
        j_center=int(round(i*slope))
        j_lo=max(0,j_center-radius); j_hi=min(M-1,j_center+radius)
        for j in range(j_lo,j_hi+1):
            cost=(query[i]-ref[j])**2
            # Soft anchor constraint: Gaussian bonus at known positions
            for k in range(len(anchor_i)):
                if anchor_i[k]==i:
                    diff=float(j-anchor_j[k])
                    cost -= 2.0 * inv2s2 * diff * diff  # -log N(j|anchor_j,sigma^2) * scale
            if i==0 and j==0: D[i,j]=cost
            elif i==0:
                prev=D[i,j-1]; D[i,j]=cost+(prev if prev<INF else INF)
            elif j==0:
                prev=D[i-1,j]; D[i,j]=cost+(prev if prev<INF else INF)
            else:
                a=D[i-1,j-1]; b=D[i-1,j]; c=D[i,j-1]
                mn=a if a<b else b; mn=mn if mn<c else c
                D[i,j]=cost+(mn if mn<INF else INF)
    i=N-1; j=M-1
    pi=np.zeros(N+M,np.int64); pj=np.zeros(N+M,np.int64); kk=0
    while i>0 or j>0:
        pi[kk]=i; pj[kk]=j; kk+=1
        if i==0: j-=1
        elif j==0: i-=1
        else:
            a=D[i-1,j-1]; b=D[i-1,j]; c=D[i,j-1]
            if a<=b and a<=c: i-=1; j-=1
            elif b<=c: i-=1
            else: j-=1
    pi[kk]=0; pj[kk]=0; kk+=1
    return D, pi[:kk], pj[:kk]

@njit(cache=True)
def _dtw_path_to_tvt(pi, pj, tw_tvt, N):
    j_for_i=np.zeros(N,np.int64)
    for k in range(len(pi)): j_for_i[pi[k]]=pj[k]
    result=np.empty(N,np.float32)
    for i in range(N): result[i]=tw_tvt[j_for_i[i]]
    return result

@njit(cache=True)
def _dtw_path_slope(pi, pj, N, smooth_win=5):
    j_for_i=np.zeros(N,np.float64)
    for k in range(len(pi)): j_for_i[pi[k]]=float(pj[k])
    slope=np.zeros(N,np.float32); hw=smooth_win//2
    for i in range(N):
        i0=max(0,i-hw); i1=min(N-1,i+hw)
        if i1>i0: slope[i]=float((j_for_i[i1]-j_for_i[i0])/(i1-i0))
        else: slope[i]=1.0
    return slope

@njit(cache=True)
def _dtw_stochastic_realizations(query, ref, radius, K, temperature):
    N=len(query); M=len(ref); INF=1e18
    slope=(M-1.0)/max(N-1.0,1.0)
    D_base=np.full((N,M),INF)
    for i in range(N):
        j_center=int(round(i*slope))
        j_lo=max(0,j_center-radius); j_hi=min(M-1,j_center+radius)
        for j in range(j_lo,j_hi+1):
            D_base[i,j]=(query[i]-ref[j])**2
    paths=np.zeros((K,N),np.int64)
    for k in range(K):
        D=np.full((N,M),INF)
        for i in range(N):
            j_center=int(round(i*slope))
            j_lo=max(0,j_center-radius); j_hi=min(M-1,j_center+radius)
            for j in range(j_lo,j_hi+1):
                noise=-temperature*np.log(-np.log(np.random.uniform(1e-10,1.0)))
                cost=D_base[i,j]+noise
                if i==0 and j==0: D[i,j]=cost
                elif i==0:
                    prev=D[i,j-1]; D[i,j]=cost+(prev if prev<INF else INF)
                elif j==0:
                    prev=D[i-1,j]; D[i,j]=cost+(prev if prev<INF else INF)
                else:
                    a=D[i-1,j-1]; b=D[i-1,j]; c=D[i,j-1]
                    mn=a if a<b else b; mn=mn if mn<c else c
                    D[i,j]=cost+(mn if mn<INF else INF)
        i=N-1; j=M-1; j_for_i=np.zeros(N,np.int64)
        while i>0 or j>0:
            j_for_i[i]=j
            if i==0: j-=1
            elif j==0: i-=1
            else:
                a=D[i-1,j-1]; b=D[i-1,j]; c=D[i,j-1]
                if a<=b and a<=c: i-=1; j-=1
                elif b<=c: i-=1
                else: j-=1
        j_for_i[0]=j; paths[k]=j_for_i
    return paths

# ── JIT warmup ──
print('JIT warmup...', flush=True)
_md=np.linspace(1,50,20,np.float64); _z=np.zeros(20,np.float64)
_gr=np.full(20,50.,np.float64); _gg=np.linspace(45,55,100,np.float64)
_pf_ancc(_md,_z,_gr,_gg,45.,0.1,20.,50.,0.,8,0.998,0.002,0.005,0.3,0.1,0.001,0.5)
_pf_ancc_with_lik(_md,_z,_gr,_gg,45.,0.1,20.,50.,0.,8,0.998,0.002,0.005,0.3,0.1,0.001,0.5,np.int64(42))
_ggs=np.linspace(45,55,100,np.float64); _grsm=np.full(20,50.,np.float64)
_pf_z(_md,_z,_gr,_grsm,_gg,_ggs,45.,0.1,20.,50.,0.,1.,0.,0.05,8,0.993,0.005,0.01,0.3,0.2,0.003,0.5)
_beam_jit(np.random.randn(30),np.random.randn(50),25,8,15.,100.)
_q=np.random.randn(40).astype(np.float64); _r=np.random.randn(50).astype(np.float64)
_dtw_sakoe_chiba(_q,_r,10)
_dtw_stochastic_realizations(_q,_r,10,3,1.0)
_anc_i=np.array([5,10],np.int64); _anc_j=np.array([6,11],np.int64)
_dtw_anchor_constrained(_q,_r,10,_anc_i,_anc_j,5.0)
print('JIT warmup done', flush=True)

# ─────────────────────────── Spatial classes ──────────────────
class FormationPlaneKNN:
    """Global-trend + local-residual-IDW² formation depth imputer (v11 improved).
    Global linear trend from ALL training wells + IDW² on residuals + stratigraphic constraints.
    LOOCV: 21.6 ft mean RMSE vs 27.3 ft for old local-plane approach.
    """
    def __init__(self, well_ids, data_dir):
        rows = []
        for wid in well_ids:
            p = data_dir / f'{wid}__horizontal_well.csv'
            try:
                df = pd.read_csv(p, usecols=['X','Y']+FORMATIONS).dropna()
            except: continue
            if len(df) == 0: continue
            row = {'wid': wid, 'x': float(df['X'].median()), 'y': float(df['Y'].median())}
            for c in FORMATIONS: row[f'{c}_m'] = float(df[c].median())
            rows.append(row)
        self.df = pd.DataFrame(rows)
        self.wmap = {w: i for i,w in enumerate(self.df['wid'])}
        xy = self.df[['x','y']].to_numpy()
        self.scale = np.where(xy.std(0) < 1e-3, 1., xy.std(0))
        self.tree = cKDTree(xy / self.scale)
        self.xa = self.df['x'].to_numpy(); self.ya = self.df['y'].to_numpy()
        self.fa = self.df[[f'{c}_m' for c in FORMATIONS]].to_numpy(np.float64)
        xy_all = np.column_stack([self.xa, self.ya, np.ones(len(self.xa))])
        self.global_coef, _, _, _ = np.linalg.lstsq(xy_all, self.fa, rcond=None)
        self.fa_resid = self.fa - xy_all @ self.global_coef
        self.strat_deltas = np.zeros(len(FORMATIONS) - 1)
        for i in range(len(FORMATIONS) - 1):
            self.strat_deltas[i] = float(np.median(self.fa[:, i+1] - self.fa[:, i]))
        print(f'  FormationPlaneKNN: {len(self.df)} wells, global trend fitted', flush=True)

    def impute(self, xy_q, self_wid=None, k=PLANE_K):
        q = xy_q / self.scale
        nf = min(k+5, len(self.df))
        dist, idx = self.tree.query(q, k=nf, workers=1)
        if self_wid in self.wmap:
            dist = np.where(idx == self.wmap[self_wid], np.inf, dist)
        ord_ = np.argpartition(dist, min(k-1,nf-1), 1)[:, :k]
        dk = np.take_along_axis(dist, ord_, 1)
        ik = np.take_along_axis(idx, ord_, 1)
        vk = np.isfinite(dk)
        w = np.where(vk, 1./(dk**2+1e-6), 0.).astype(np.float64)
        sw = w.sum(1, keepdims=True); sw = np.where(sw < 1e-9, 1., sw)
        wn = w / sw
        Xq_aug = np.column_stack([xy_q, np.ones(len(xy_q))])
        global_pred = (Xq_aug @ self.global_coef).astype(np.float64)
        resid_k = self.fa_resid[ik]
        resid_pred = (resid_k * wn[:, :, None]).sum(1)
        pred = (global_pred + resid_pred).astype(np.float32)
        for i in range(len(FORMATIONS) - 1):
            pred[:, i+1] = np.minimum(pred[:, i+1], pred[:, i] + self.strat_deltas[i] * 0.5)
        pred[~vk.any(1)] = self.fa.mean(0).astype(np.float32)
        return pred, np.where(vk, dk, np.inf).min(1).astype(np.float32)


class DenseANCCImputer:
    """Global-trend + local-residual-IDW² for dense ANCC imputation (v11 improved).
    LOOCV: 12.9 ft mean RMSE.
    """
    def __init__(self, well_ids, data_dir, spw=DENSE_SPW):
        xs,ys,anccs,wids=[],[],[],[]
        for wid in well_ids:
            p = data_dir / f'{wid}__horizontal_well.csv'
            try: df = pd.read_csv(p, usecols=['X','Y','ANCC']).dropna()
            except: continue
            if len(df)==0: continue
            ix=np.linspace(0,len(df)-1,min(spw,len(df)),dtype=int); s=df.iloc[ix]
            xs.append(s['X'].values); ys.append(s['Y'].values)
            anccs.append(s['ANCC'].values); wids.extend([wid]*len(s))
        self.xy=np.column_stack([np.concatenate(xs),np.concatenate(ys)])
        self.ancc=np.concatenate(anccs).astype(np.float64)
        self.wids=np.array(wids)
        self.scale=np.where(self.xy.std(0)<1e-3,1.,self.xy.std(0))
        self.tree=cKDTree(self.xy/self.scale)
        xy_aug = np.column_stack([self.xy, np.ones(len(self.xy))])
        self.global_coef, _, _, _ = np.linalg.lstsq(xy_aug, self.ancc, rcond=None)
        self.ancc_resid = (self.ancc - xy_aug @ self.global_coef).astype(np.float32)
        self.ancc = self.ancc.astype(np.float32)
        print(f'  DenseANCCImputer: {len(self.ancc)} points, global trend fitted', flush=True)

    def impute(self, xy_q, self_wid=None, k=DENSE_K, nfetch=5000):
        xy_q=np.atleast_2d(xy_q); q=xy_q/self.scale
        nf=min(nfetch,len(self.ancc))
        dist,idx=self.tree.query(q,k=nf,workers=1)
        if self_wid: dist=np.where(self.wids[idx]==self_wid,np.inf,dist)
        ord_=np.argpartition(dist,min(k-1,nf-1),1)[:,:k]
        dk=np.take_along_axis(dist,ord_,1); ik=np.take_along_axis(idx,ord_,1)
        vk=np.isfinite(dk)
        w=np.where(vk,1./(dk**2+1e-6),0.)
        sw=w.sum(1); safe=np.where(sw<1e-9,1.,sw)
        Xq_aug = np.column_stack([xy_q, np.ones(len(xy_q))])
        global_pred = (Xq_aug @ self.global_coef).astype(np.float32)
        resid_k = self.ancc_resid[ik]
        resid_pred = (resid_k * w).sum(1) / safe
        resid_pred = np.where(sw < 1e-9, 0., resid_pred).astype(np.float32)
        ap = global_pred + resid_pred
        ap = np.where(sw < 1e-9, float(self.ancc.mean()), ap)
        var=((resid_k - resid_pred[:,None])**2 * w).sum(1)/safe
        return ap.astype(np.float32), np.sqrt(np.maximum(var,0.)).astype(np.float32), \
               np.where(vk,dk,np.inf).min(1).astype(np.float32)

# ─────────────────────────── Python helpers ───────────────────
def list_wells(data_dir):
    return sorted({f.name.replace('__horizontal_well.csv','')
                   for f in data_dir.glob('*__horizontal_well.csv')})

def load_well(wid, data_dir):
    hw = pd.read_csv(data_dir/f'{wid}__horizontal_well.csv')
    tw = pd.read_csv(data_dir/f'{wid}__typewell.csv').sort_values('TVT')
    return hw, tw

def _grid(tw_tvt, tw_gr, step=0.2):
    tmin=float(tw_tvt.min()); tmax=float(tw_tvt.max())
    tvt_g=np.arange(tmin,tmax+step,step)
    return np.interp(tvt_g,tw_tvt,tw_gr).astype(np.float64),float(tmin),float(step)

def _gr_sig(hw, tw_tvt, tw_gr):
    kn=hw[hw['TVT_input'].notna()&hw['GR'].notna()]
    if len(kn)<20: return float(PF_GR_SIG_DEF)
    return float(np.clip(np.std(kn['GR'].values-np.interp(kn['TVT_input'].values,tw_tvt,tw_gr)),
                          PF_GR_SIG_MIN,PF_GR_SIG_MAX))

def _nn(arr, v):
    i=int(np.searchsorted(arr,v,'left'))
    if i>=len(arr): return len(arr)-1
    if i>0 and abs(arr[i-1]-v)<=abs(arr[i]-v): return i-1
    return i

def _smooth(vals, fb, r):
    s=pd.Series(vals,dtype='float32').interpolate(limit_direction='both').fillna(fb)
    return (s.rolling(r*2+1,center=True,min_periods=1).mean() if r>0 else s).to_numpy(np.float32)

def robust_slope(x, y):
    x=np.asarray(x,float); y=np.asarray(y,float)
    m=np.isfinite(x)&np.isfinite(y)
    if m.sum()<2 or np.std(x[m])<1e-6: return 0.
    return float(np.polyfit(x[m],y[m],1)[0])

def affine_cal(kgr, tw_at_k, min_pts=20):
    v=np.isfinite(kgr)&np.isfinite(tw_at_k)
    if v.sum()<min_pts or np.std(tw_at_k[v])<1e-6:
        return 1., float(np.nanmean(kgr)-np.nanmean(tw_at_k)) if v.any() else 0.
    a,b=np.polyfit(tw_at_k[v],kgr[v],1); return float(a),float(b)

def seg_b_well(ktvt, kz, form_col):
    bv=ktvt+kz-form_col; n=len(bv)
    b_full=float(np.median(bv))
    b_late=float(np.median(bv[max(0,n-50):])) if n>=5 else b_full
    t1,t2=n//3,2*n//3
    b_early=float(np.median(bv[:max(1,t1)])) if t1>0 else b_full
    b_mid=float(np.median(bv[t1:max(t1+1,t2)])) if t2>t1 else b_full
    wts=np.exp(0.02*np.arange(n)); wts/=wts.sum()
    b_wls=float(np.dot(wts,bv))
    return b_full,b_early,b_mid,b_late,b_wls

def beam_search(gr_h, tw_tvt, tw_gr, start_tvt, bs, mc, es, r):
    si=_nn(tw_tvt,start_tvt)
    sgr=_smooth(gr_h,float(np.nanmean(tw_gr)),r).astype(np.float64)
    path=_beam_jit(sgr,tw_gr.astype(np.float64),si,bs,float(mc),float(es))
    return tw_tvt[path].astype(np.float32)

def run_pf_ancc(hw, tw_tvt, tw_gr, N=PF_N):
    gs=_gr_sig(hw,tw_tvt,tw_gr)
    kn=hw[hw['TVT_input'].notna()]; ev=hw[hw['TVT_input'].isna()]
    if len(ev)==0: return np.array([]),np.array([])
    ls=float(kn['TVT_input'].iloc[-1]+kn['Z'].iloc[-1])
    tail=kn.tail(30); dt=np.diff(tail['TVT_input'].values); dz=np.diff(tail['Z'].values)
    dm=np.diff(tail['MD'].values); m=dm>0
    ir=float(np.median((dt+dz)[m]/dm[m])) if m.sum()>=3 else 0.
    gg,gmin,gst=_grid(tw_tvt,tw_gr)
    pts,std=_pf_ancc(ev['MD'].values.astype(np.float64),ev['Z'].values.astype(np.float64),
                     ev['GR'].values.astype(np.float64),gg,gmin,gst,
                     gs,ls,ir,N,ANCC_ALPHA,ANCC_RN,ANCC_PN,ANCC_IS,ANCC_RP,ANCC_RR,PF_RESAMP)
    return pts.astype(np.float32),std.astype(np.float32)

def run_pf_likelihood_ensemble(hw, tw_tvt, tw_gr, K=128,
                                temps=(3., 5., 8., 12.), N=500):
    """128-seed likelihood-weighted PF ensemble. Returns 6 delta features per unknown row.
    N=500 matches fle3n (was 100 before, caused noisy estimates).
    """
    gs = _gr_sig(hw, tw_tvt, tw_gr)
    kn = hw[hw['TVT_input'].notna()]; ev = hw[hw['TVT_input'].isna()]
    if len(ev) == 0:
        empty = np.zeros(0, np.float32)
        out = {f'pf_lik_{t:g}_d': empty for t in temps}
        out['pf_lik_mean_d'] = empty; out['pf_pt_std'] = empty
        return out
    ls = float(kn['TVT_input'].iloc[-1] + kn['Z'].iloc[-1])
    tail = kn.tail(30); dt = np.diff(tail['TVT_input'].values)
    dz = np.diff(tail['Z'].values); dm = np.diff(tail['MD'].values); m = dm > 0
    ir = float(np.median((dt + dz)[m] / dm[m])) if m.sum() >= 3 else 0.
    gg, gmin, gst = _grid(tw_tvt, tw_gr)
    md_v = ev['MD'].values.astype(np.float64)
    z_v  = ev['Z'].values.astype(np.float64)
    gr_v = ev['GR'].values.astype(np.float64)
    nh = len(ev)
    all_preds = np.empty((K, nh), dtype=np.float32)
    all_liks  = np.empty(K, dtype=np.float64)
    for k in range(K):
        pts, log_lik = _pf_ancc_with_lik(
            md_v, z_v, gr_v, gg, gmin, gst, gs, ls, ir, N,
            ANCC_ALPHA, ANCC_RN, ANCC_PN, ANCC_IS, ANCC_RP, ANCC_RR, PF_RESAMP,
            np.int64(k * 7 + 42))
        all_preds[k] = pts.astype(np.float32)  # pts IS already TVT (= pos - z weighted mean)
        all_liks[k]  = log_lik
    # Convert absolute TVT predictions to deltas (TVT - last_known_tvt)
    # This matches fle3n: likpf_scale_5_d = likpf_scale_5 - last_known_tvt
    last_tvt_val = float(kn['TVT_input'].iloc[-1])
    all_deltas = (all_preds - last_tvt_val).astype(np.float32)
    out = {}
    out['pf_lik_mean_d'] = all_deltas.mean(0).astype(np.float32)
    out['pf_pt_std']     = all_deltas.std(0).astype(np.float32)
    ln = all_liks - all_liks.max()
    for temp in temps:
        wts = np.exp(ln / temp); wts /= wts.sum()
        out[f'pf_lik_{temp:g}_d'] = (wts[:, None] * all_deltas).sum(0).astype(np.float32)
    return out

def run_pf_z(hw, tw_tvt, tw_gr, N=PF_N):
    gs=_gr_sig(hw,tw_tvt,tw_gr)
    tw_s=pd.Series(tw_gr).rolling(PF_GR_WIN,center=True,min_periods=1).mean().values.astype(np.float32)
    kna=hw[hw['TVT_input'].notna()]; ev=hw[hw['TVT_input'].isna()]
    if len(ev)==0: return np.array([]),np.array([])
    dz_k=np.diff(kna['Z'].values); dvt=np.diff(kna['TVT_input'].values)
    dmd_k=np.diff(kna['MD'].values); m2=dmd_k>0
    if m2.sum()>=10:
        vz=dz_k[m2]/dmd_k[m2]; vt=dvt[m2]/dmd_k[m2]
        A=np.column_stack([vz,np.ones_like(vz)]); c,_,_,_=np.linalg.lstsq(A,vt,rcond=None)
        beta,icpt,zsig=float(c[0]),float(c[1]),max(float(np.std(vt-(c[0]*vz+c[1]))),0.001)
    else:
        beta,icpt,zsig=-1.,0.,0.1
    t2=kna.tail(20); dvt2=np.diff(t2['TVT_input'].values); dmd2=np.diff(t2['MD'].values); m3=dmd2>0
    iv=float(np.median(dvt2[m3]/dmd2[m3])) if m3.sum()>=3 else 0.
    gg,gmin,gst=_grid(tw_tvt,tw_gr)
    ggs,_,_=_grid(tw_tvt,tw_s)
    gr_sm=hw['GR'].rolling(PF_GR_WIN,center=True,min_periods=1).mean()
    pts,std=_pf_z(ev['MD'].values.astype(np.float64),ev['Z'].values.astype(np.float64),
                  ev['GR'].values.astype(np.float64),
                  gr_sm.loc[ev.index].values.astype(np.float64),
                  gg,ggs,gmin,gst,gs,float(kna['TVT_input'].iloc[-1]),iv,
                  beta,icpt,zsig,N,
                  PF_MOM,PF_VN,PF_PN,PF_GR_WT,PF_ROUGH_P,PF_ROUGH_V,PF_RESAMP)
    return pts.astype(np.float32),std.astype(np.float32)

def run_dtw_anchor_constrained(full_gr, tw_tvt, tw_gr, hw, radii=DTW_RADII, sigma=8.0):
    """Anchor-constrained DTW using known TVT_input positions as soft Gaussian constraints.

    Incorporates the ~25% known TVT anchors directly into the DTW cost matrix,
    biasing the warp path through geologically correct positions.
    Returns ensemble TVT and per-radius results for the full signal.
    """
    N = len(full_gr)
    qn = ((full_gr - full_gr.mean()) / (full_gr.std() + 1e-6)).astype(np.float64)
    rn = ((tw_gr - tw_gr.mean()) / (tw_gr.std() + 1e-6)).astype(np.float64)

    # Build anchor arrays: known anchor positions in full GR signal → typewell indices
    kn = hw[hw['TVT_input'].notna() & hw['GR'].notna()]
    anchor_i_list = []
    anchor_j_list = []
    for idx in kn.index:
        # Position in full GR array
        i = int(np.searchsorted(hw.index.values, idx))
        i = min(max(i, 0), N - 1)
        # Corresponding typewell index (nearest TVT in typewell)
        tvt_val = float(kn.loc[idx, 'TVT_input'])
        j = int(np.argmin(np.abs(tw_tvt - tvt_val)))
        j = min(max(j, 0), len(tw_tvt) - 1)
        anchor_i_list.append(i)
        anchor_j_list.append(j)

    anchor_i = np.array(anchor_i_list, dtype=np.int64)
    anchor_j = np.array(anchor_j_list, dtype=np.int64)

    # Run constrained DTW at each radius
    dtw_tvts = {}; dtw_costs = {}; inv_sum = 0.; tvt_stack = []
    for r in radii:
        D, pi, pj = _dtw_anchor_constrained(qn, rn, r, anchor_i, anchor_j, sigma)
        cost = float(D[N-1, len(rn)-1]) / max(N + len(rn), 1)
        tvt_pred = _dtw_path_to_tvt(pi[::-1], pj[::-1], tw_tvt.astype(np.float32), N)
        dtw_tvts[r] = tvt_pred; dtw_costs[r] = cost
        ic = 1. / (abs(cost) + 1e-6); inv_sum += ic; tvt_stack.append((tvt_pred, ic))

    weights = np.array([ic / inv_sum for _, ic in tvt_stack], dtype=np.float32)
    tvts_mat = np.stack([t for t, _ in tvt_stack], axis=1)
    dtw_anc_ens = (tvts_mat * weights[None, :]).sum(axis=1).astype(np.float32)
    return dtw_tvts, dtw_costs, dtw_anc_ens

def run_dtw_multiscale(query_gr, tw_tvt, tw_gr, radii=DTW_RADII):
    N=len(query_gr)
    qn=((query_gr-query_gr.mean())/(query_gr.std()+1e-6)).astype(np.float64)
    rn=((tw_gr-tw_gr.mean())/(tw_gr.std()+1e-6)).astype(np.float64)
    dtw_tvts={}; dtw_slopes={}; dtw_costs={}; inv_sum=0.; tvt_stack=[]
    for r in radii:
        D,pi,pj=_dtw_sakoe_chiba(qn,rn,r)
        cost=float(D[N-1,len(rn)-1])/max(N+len(rn),1)
        tvt_pred=_dtw_path_to_tvt(pi[::-1],pj[::-1],tw_tvt.astype(np.float32),N)
        slope=_dtw_path_slope(pi[::-1],pj[::-1],N)
        dtw_tvts[r]=tvt_pred; dtw_slopes[r]=slope; dtw_costs[r]=cost
        ic=1./(cost+1e-6); inv_sum+=ic; tvt_stack.append((tvt_pred,ic))
    weights=np.array([ic/inv_sum for _,ic in tvt_stack],dtype=np.float32)
    tvts_mat=np.stack([t for t,_ in tvt_stack],axis=1)
    dtw_ens=(tvts_mat*weights[None,:]).sum(axis=1).astype(np.float32)
    return dtw_tvts,dtw_slopes,dtw_costs,dtw_ens

def run_dtw_deriv_multiscale(query_gr, tw_tvt, tw_gr, radii=DTW_RADII):
    """DTW on GR derivatives: detects formation boundaries (transitions) more precisely.
    Derivative DTW aligns dGR/dMD signal against dtypewell_GR/dTVT — works where
    absolute GR has calibration drift but boundary sharpness is preserved.
    """
    N = len(query_gr)
    if N < 3: return {}, np.full(N, tw_tvt[-1] if len(tw_tvt)>0 else 0., np.float32)

    # Compute derivatives (central differences, pad edges)
    dq = np.gradient(query_gr.astype(np.float64))
    dr = np.gradient(tw_gr.astype(np.float64))

    # Normalize per-signal
    qn = (dq / (np.abs(dq).mean() + 1e-6))
    rn = (dr / (np.abs(dr).mean() + 1e-6))
    qn = qn.astype(np.float64); rn = rn.astype(np.float64)

    dtw_tvts = {}; dtw_costs = {}; inv_sum = 0.; tvt_stack = []
    for r in radii:
        D, pi, pj = _dtw_sakoe_chiba(qn, rn, r)
        cost = float(D[N-1, len(rn)-1]) / max(N + len(rn), 1)
        tvt_pred = _dtw_path_to_tvt(pi[::-1], pj[::-1], tw_tvt.astype(np.float32), N)
        dtw_tvts[r] = tvt_pred; dtw_costs[r] = cost
        ic = 1. / (cost + 1e-6); inv_sum += ic; tvt_stack.append((tvt_pred, ic))

    weights = np.array([ic / inv_sum for _, ic in tvt_stack], dtype=np.float32)
    tvts_mat = np.stack([t for t, _ in tvt_stack], axis=1)
    dtw_deriv_ens = (tvts_mat * weights[None, :]).sum(axis=1).astype(np.float32)
    return dtw_tvts, dtw_costs, dtw_deriv_ens

def run_dtw_stochastic(query_gr, tw_tvt, tw_gr, radius=50, K=DTW_STOCH_K, temperature=DTW_STOCH_TEMP):
    N=len(query_gr)
    qn=((query_gr-query_gr.mean())/(query_gr.std()+1e-6)).astype(np.float64)
    rn=((tw_gr-tw_gr.mean())/(tw_gr.std()+1e-6)).astype(np.float64)
    paths=_dtw_stochastic_realizations(qn,rn,radius,K,temperature)
    tvt_realiz=np.empty((K,N),dtype=np.float32)
    for k in range(K):
        for i in range(N): tvt_realiz[k,i]=tw_tvt[paths[k,i]]
    mean_tvt=tvt_realiz.mean(axis=0).astype(np.float32)
    std_tvt=tvt_realiz.std(axis=0).astype(np.float32)
    cv_tvt=(std_tvt/(np.abs(mean_tvt)+1e-6)).astype(np.float32)
    return mean_tvt,std_tvt,cv_tvt

def multi_scale_ncc(kgr, ktvt, hgr, hws=(8,15,25), stride=3):
    out=[]
    for hw in hws:
        win=2*hw+1; nk=len(kgr); nh=len(hgr)
        if nk<win+1 or nh==0:
            out.append((np.full(nh,ktvt[-1],np.float32),np.zeros(nh,np.float32))); continue
        kg=pd.Series(kgr).rolling(5,center=True,min_periods=1).mean().values.astype(np.float32)
        hg=pd.Series(hgr).rolling(5,center=True,min_periods=1).mean().values.astype(np.float32)
        sts=np.arange(0,nk-win+1,stride,dtype=np.int32); M=len(sts)
        if M==0:
            out.append((np.full(nh,ktvt[-1],np.float32),np.zeros(nh,np.float32))); continue
        C=kg[sts[:,None]+np.arange(win,dtype=np.int32)[None,:]].astype(np.float32)
        Cn=(C-C.mean(1,keepdims=True))/(C.std(1,keepdims=True)+1e-6)
        hp=np.pad(hg,hw,mode='edge')
        H=hp[np.arange(nh)[:,None]+np.arange(win)[None,:]].astype(np.float32)
        Hn=(H-H.mean(1,keepdims=True))/(H.std(1,keepdims=True)+1e-6)
        ncc=Hn@Cn.T/win; best=ncc.argmax(1); score=ncc.max(1).astype(np.float32)
        out.append((ktvt[np.clip(sts[best]+hw,0,nk-1)].astype(np.float32),score))
    tvts=np.stack([o[0] for o in out],1); scores=np.stack([o[1] for o in out],1)
    sw=np.exp(3.*scores); sw/=sw.sum(1,keepdims=True)+1e-9
    sc_ens=(tvts*sw).sum(1).astype(np.float32)
    return out,sc_ens

# ─────────────────────────── Wavelet features ─────────────────
_WAV_WAVELET = 'db4'
_WAV_LEVEL   = 4
_WAV_WIN     = 64   # local window for wavelet features around each eval point

def _wavelet_features(gr_full, ev_start, nh):
    """
    Compute local DWT features for each evaluation point.
    For each eval point i, extract wavelet coefficients from a
    window of GR signal centered at i.
    Returns dict of arrays of shape (nh,).
    """
    N = len(gr_full)
    hw = _WAV_WIN // 2
    feats = {}
    # Per-level arrays
    for lv in range(_WAV_LEVEL + 1):
        feats[f'wav_mean_l{lv}']   = np.zeros(nh, np.float32)
        feats[f'wav_std_l{lv}']    = np.zeros(nh, np.float32)
        feats[f'wav_energy_l{lv}'] = np.zeros(nh, np.float32)
    # Dominant frequency feature
    feats['wav_dom_scale'] = np.zeros(nh, np.float32)

    for ii in range(nh):
        gi = ev_start + ii
        lo = max(0, gi - hw); hi = min(N, gi + hw)
        seg = gr_full[lo:hi].astype(np.float64)
        if len(seg) < 8:
            continue
        # Multi-level DWT decomposition
        try:
            coeffs = pywt.wavedec(seg, _WAV_WAVELET, level=_WAV_LEVEL)
        except Exception:
            continue
        energies = []
        for lv, c in enumerate(coeffs):
            e = float(np.mean(c**2)) if len(c) > 0 else 0.
            feats[f'wav_mean_l{lv}'][ii]   = float(np.mean(c))   if len(c) > 0 else 0.
            feats[f'wav_std_l{lv}'][ii]    = float(np.std(c))    if len(c) > 0 else 0.
            feats[f'wav_energy_l{lv}'][ii] = e
            energies.append(e)
        # Dominant scale = level with max energy (ignoring approx coeff l0)
        if len(energies) > 1 and sum(energies[1:]) > 0:
            feats['wav_dom_scale'][ii] = float(np.argmax(energies[1:]) + 1)
    return feats

# Also: global wavelet stats from typewell template vs query GR
def _wavelet_corr_features(gr_seg, tw_tvt, tw_gr, pred_tvt, window=80):
    """Compare local wavelet power spectrum of GR to typewell template at pred_tvt."""
    N = len(gr_seg); out = np.zeros(N, np.float32)
    hw = window // 2
    for ii in range(N):
        tvt_c = float(pred_tvt[ii]) if hasattr(pred_tvt, '__len__') else float(pred_tvt)
        # Query segment
        lo = max(0, ii - hw); hi = min(N, ii + hw)
        q_seg = gr_seg[lo:hi].astype(np.float64)
        # Template segment from typewell around pred_tvt
        tw_lo = max(0, np.searchsorted(tw_tvt, tvt_c - 40))
        tw_hi = min(len(tw_tvt), np.searchsorted(tw_tvt, tvt_c + 40))
        if tw_hi - tw_lo < 8 or len(q_seg) < 8:
            continue
        t_seg = tw_gr[tw_lo:tw_hi].astype(np.float64)
        # Power spectral similarity (ratio of energies per level)
        try:
            qc = pywt.wavedec(q_seg, 'db4', level=3)
            tc = pywt.wavedec(t_seg, 'db4', level=3)
            q_e = np.array([np.mean(c**2) if len(c) else 0. for c in qc])
            t_e = np.array([np.mean(c**2) if len(c) else 0. for c in tc])
            sim = float(np.dot(q_e, t_e) / (np.linalg.norm(q_e) * np.linalg.norm(t_e) + 1e-9))
            out[ii] = sim
        except Exception:
            pass
    return out

# ─────────────────────────── build_well ───────────────────────
_FI = None  # FormationPlaneKNN (initialized in main)
_DI = None  # DenseANCCImputer  (initialized in main)

def build_well(wid, data_dir, is_train=True):
    global _FI, _DI
    try:
        hw, tw = load_well(wid, data_dir)
    except: return None

    kn=hw[hw['TVT_input'].notna()]; ev=hw[hw['TVT_input'].isna()]
    if len(ev)==0 or len(kn)<10: return None
    if is_train and ('TVT' not in hw.columns or hw['TVT'].isna().all()): return None

    tw_tvt=tw['TVT'].to_numpy(np.float32); tw_gr=tw['GR'].to_numpy(np.float32)
    if len(tw_tvt)<3: return None

    lk=kn.iloc[-1]; last_tvt=float(lk['TVT_input'])
    gr_full=hw['GR'].astype(float).interpolate(limit_direction='both').fillna(float(np.nanmean(tw_gr))).values.astype(np.float32)
    hgr=gr_full[ev.index[0]:].astype(np.float32)
    kgr=gr_full[:len(kn)].astype(np.float32)
    ktvt=kn['TVT_input'].to_numpy(np.float32)
    nh=len(ev); ev_start=ev.index[0]

    # PF ANCC
    pf_a, std_a = run_pf_ancc(hw, tw_tvt, tw_gr)
    if len(pf_a)==0: return None

    # 128-seed likelihood-weighted PF ensemble (fle3n key feature)
    pf_lik_feats = run_pf_likelihood_ensemble(hw, tw_tvt, tw_gr)

    # PF Z
    pf_z_pts, std_z = run_pf_z(hw, tw_tvt, tw_gr)
    has_z = len(pf_z_pts)==len(pf_a) and not np.any(np.isnan(pf_z_pts))

    # Beam search (7 configs)
    bpaths={}
    for (bs,mc,es,r,tag) in BEAMS:
        bpaths[tag]=beam_search(hgr,tw_tvt,tw_gr,last_tvt,bs,mc,es,r)
    beam_ref=(bpaths['cons']+bpaths['sm5'])/2.

    # NCC
    sc_res,sc_ens=multi_scale_ncc(kgr,ktvt,hgr[:nh],hws=(8,15,25),stride=3)
    sc8,sc8s=sc_res[0]; sc15,sc15s=sc_res[1]; sc25,sc25s=sc_res[2]
    sc_cons=(sc8+sc15+sc25)/3.
    sc_trust=float(np.clip(len(kn)/200.,0.,0.6))
    hyb_ref=(1-sc_trust)*beam_ref+sc_trust*sc_ens

    # DTW multi-scale
    dtw_tvts,dtw_slopes,dtw_costs,dtw_ens=run_dtw_multiscale(gr_full,tw_tvt,tw_gr)
    def _ev(arr): return arr[ev_start:ev_start+nh].astype(np.float32)
    dtw_ens_ev=_ev(dtw_ens)
    dtw_per_r={r: _ev(dtw_tvts[r]) for r in DTW_RADII}
    dtw_slope_ev={r: _ev(dtw_slopes[r]) for r in DTW_RADII}

    # DTW stochastic
    dtw_mean_stoch,dtw_std_stoch,dtw_cv_stoch=run_dtw_stochastic(gr_full,tw_tvt,tw_gr)
    dtw_mean_ev=_ev(dtw_mean_stoch); dtw_std_ev=_ev(dtw_std_stoch); dtw_cv_ev=_ev(dtw_cv_stoch)

    # Derivative DTW: aligns dGR/dMD against dtypewell_GR/dTVT
    dtw_deriv_tvts, dtw_deriv_costs, dtw_deriv_ens = run_dtw_deriv_multiscale(gr_full, tw_tvt, tw_gr)
    dtw_deriv_ens_ev = _ev(dtw_deriv_ens)

    # Anchor-constrained DTW: uses known TVT_input anchors as soft Gaussian constraints
    # biasing the warp path through geologically correct typewell positions
    dtw_anc_tvts, dtw_anc_costs, dtw_anc_ens = run_dtw_anchor_constrained(gr_full, tw_tvt, tw_gr, hw)
    dtw_anc_ens_ev = _ev(dtw_anc_ens)
    dtw_anc_per_r = {r: _ev(dtw_anc_tvts[r]) for r in DTW_RADII}
    dtw_anc_cost_min = float(min(dtw_anc_costs.values()))
    dtw_deriv_per_r = {r: _ev(dtw_deriv_tvts[r]) for r in DTW_RADII}
    # Cross-signal cost: low = derivative alignment is confident
    dtw_deriv_cost_min = float(min(dtw_deriv_costs.values()))

    # Affine cal
    tw_at_k=np.interp(ktvt,tw_tvt,tw_gr).astype(np.float32)
    a_cal,b_cal=affine_cal(kgr,tw_at_k)
    pfx_rmse=float(np.sqrt(np.mean((kgr-tw_at_k)**2)))

    # Slope features
    slp_all=robust_slope(kn['MD'].values,ktvt)
    slp_50=robust_slope(kn['MD'].values[-50:],ktvt[-50:])
    slp_z=robust_slope(kn['Z'].values,ktvt)
    md_ev=ev['MD'].to_numpy(np.float32); z_ev=ev['Z'].to_numpy(np.float32)
    md_since=md_ev-float(lk['MD'])
    slp_b_all=(last_tvt+slp_all*md_since).astype(np.float32)
    slp_b_50=(last_tvt+slp_50*md_since).astype(np.float32)

    frac=(np.arange(nh)/max(nh-1,1)).astype(np.float32)
    dzdmd_all=np.gradient(hw['Z'].values,hw['MD'].values)
    dzdmd=dzdmd_all[ev_start:ev_start+nh].astype(np.float32)
    dxdmd=np.gradient(hw['X'].values,hw['MD'].values)[ev_start:ev_start+nh].astype(np.float32)
    dydmd=np.gradient(hw['Y'].values,hw['MD'].values)[ev_start:ev_start+nh].astype(np.float32)

    hgr_ev=gr_full[ev_start:ev_start+nh]
    tw_interp=interp1d(tw_tvt,tw_gr,bounds_error=False,fill_value='extrapolate')
    gr_s=pd.Series(gr_full)
    gr_d1=gr_s.diff().fillna(0.).values[ev_start:ev_start+nh].astype(np.float32)
    gr_d2=gr_s.diff().diff().fillna(0.).values[ev_start:ev_start+nh].astype(np.float32)
    gr_env=gr_s.rolling(21,center=True,min_periods=1).max().values[ev_start:ev_start+nh].astype(np.float32)
    gr_nrg=np.sqrt(np.maximum((gr_s**2).rolling(21,center=True,min_periods=1).mean(),0.)).values[ev_start:ev_start+nh].astype(np.float32)

    def sc(v): return np.full(nh,np.float32(v),np.float32)

    # ── Spatial formation features ──
    swid=wid if is_train else None
    xy_ev=ev[['X','Y']].to_numpy(np.float64); xy_kn=kn[['X','Y']].to_numpy(np.float64)
    form_ev,knn_d=_FI.impute(xy_ev,self_wid=swid)
    form_kn,_=_FI.impute(xy_kn,self_wid=swid)
    z_kn=kn['Z'].to_numpy(np.float32)

    tvt_fs={}; form_rmse={}
    form_list=[]
    for fi2,fn in enumerate(FORMATIONS):
        b_full,b_early,b_mid,b_late,b_wls=seg_b_well(ktvt,z_kn,form_kn[:,fi2])
        tvt_f=(-z_ev+form_ev[:,fi2]+b_full).astype(np.float32)
        tvt_fw=(-z_ev+form_ev[:,fi2]+b_wls).astype(np.float32)
        tvt_f50=(-z_ev+form_ev[:,fi2]+b_late).astype(np.float32)
        tvt_fs[f'tvtF_{fn}']=tvt_f; tvt_fs[f'tvtFw_{fn}']=tvt_fw
        tvt_fs[f'tvtF50_{fn}']=tvt_f50
        tvt_fs[f'bw_{fn}']=sc(b_full); tvt_fs[f'bww_{fn}']=sc(b_wls)
        tvt_fs[f'bw50_{fn}']=sc(b_late)
        form_rmse[fn]=float(np.sqrt(np.mean((ktvt-(-z_kn+form_kn[:,fi2]+b_full))**2)))
        form_list.append(tvt_f)
    fs=np.stack(form_list,1)
    form_mean_d=(fs.mean(1)-last_tvt).astype(np.float32)
    form_std_d=fs.std(1).astype(np.float32)
    form_rng_d=(fs.max(1)-fs.min(1)).astype(np.float32)

    # ── Dense ANCC features ──
    d_ancc,d_std,d_dist=_DI.impute(xy_ev,self_wid=swid)
    d_kn,d_std_kn,_=_DI.impute(xy_kn,self_wid=swid)
    b_vd=ktvt+z_kn-d_kn
    _,b_de,b_dm,b_dl,b_dw=seg_b_well(ktvt,z_kn,d_kn)
    b_d=float(np.median(b_vd))
    tvt_dense=(-z_ev+d_ancc+b_d).astype(np.float32)
    tvt_densew=(-z_ev+d_ancc+b_dw).astype(np.float32)
    tvt_dense50=(-z_ev+d_ancc+b_dl).astype(np.float32)
    d_rmse=float(np.sqrt(np.mean((ktvt+z_kn-d_kn)**2)))
    d_bias=float(np.mean(b_vd)); d_nb_std=float(np.mean(d_std_kn))

    # Signal spread
    all_sigs=[pf_a]+[p for p in bpaths.values()]+[sc8,sc15,sc25,sc_ens,tvt_fs['tvtF_ANCC'],tvt_dense,dtw_ens_ev,dtw_deriv_ens_ev,dtw_anc_ens_ev]
    sig_mat=np.stack(all_sigs,1)
    sig_std=sig_mat.std(1).astype(np.float32)
    sig_mean=(sig_mat.mean(1)-last_tvt).astype(np.float32)

    # DTW cost stats
    dtw_cost_arr=np.array([dtw_costs[r] for r in DTW_RADII],dtype=np.float32)
    dtw_cost_min=float(dtw_cost_arr.min()); dtw_cost_range=float(dtw_cost_arr.max()-dtw_cost_arr.min())

    feats = {
        'well': wid, 'id': [f'{wid}_{i}' for i in ev.index],
        'last_known_tvt': sc(last_tvt),
        # PF ANCC
        'pf_ancc': pf_a, 'pf_ancc_std': std_a,
        'pf_ancc_delta': (pf_a-last_tvt).astype(np.float32),
        # 128-seed likelihood-weighted PF ensemble (fle3n key feature)
        **pf_lik_feats,
        # PF Z
        'pf_z': (pf_z_pts.astype(np.float32) if has_z else sc(last_tvt)),
        'pf_z_delta': ((pf_z_pts-last_tvt).astype(np.float32) if has_z else sc(0.)),
        'pf_vs_z': ((pf_a-pf_z_pts.astype(np.float32)) if has_z else sc(0.)),
        # Beams
        **{f'beam_{t}_d': (p-np.float32(last_tvt)).astype(np.float32) for t,p in bpaths.items()},
        'beam_mean_d': np.stack([(p-last_tvt) for p in bpaths.values()],1).mean(1).astype(np.float32),
        'beam_std_d':  np.stack([(p-last_tvt) for p in bpaths.values()],1).std(1).astype(np.float32),
        'beam_med_d':  np.median(np.stack([(p-last_tvt) for p in bpaths.values()],1),1).astype(np.float32),
        # NCC
        'sc8_d': (sc8-np.float32(last_tvt)).astype(np.float32), 'sc8_sc': sc8s,
        'sc15_d': (sc15-np.float32(last_tvt)).astype(np.float32), 'sc15_sc': sc15s,
        'sc25_d': (sc25-np.float32(last_tvt)).astype(np.float32), 'sc25_sc': sc25s,
        'sc_cons_d': (sc_cons-np.float32(last_tvt)).astype(np.float32),
        'sc_ens_d': (sc_ens-np.float32(last_tvt)).astype(np.float32),
        'sc_trust': sc(sc_trust),
        'hyb_d': (hyb_ref-np.float32(last_tvt)).astype(np.float32),
        # Signal spread
        'sig_std': sig_std, 'sig_mean_d': sig_mean,
        # Formation plane features
        **tvt_fs,
        **{f'frm_rmse_{fn}': sc(form_rmse[fn]) for fn in FORMATIONS},
        'form_mean_d': form_mean_d, 'form_std_d': form_std_d, 'form_rng_d': form_rng_d,
        'spatial_knn_dist': knn_d,
        # Dense ANCC
        'dense_ancc': d_ancc, 'dense_std': d_std, 'dense_dist': d_dist,
        'tvt_dense_d': (tvt_dense-last_tvt).astype(np.float32),
        'tvt_densew_d': (tvt_densew-last_tvt).astype(np.float32),
        'tvt_dense50_d': (tvt_dense50-last_tvt).astype(np.float32),
        'dense_rmse': sc(d_rmse), 'dense_bias': sc(d_bias), 'dense_nb_std': sc(d_nb_std),
        # Cross-signal comparisons
        'pf_vs_spatial': (pf_a-tvt_fs['tvtF_ANCC']).astype(np.float32),
        'pf_vs_dense': (pf_a-tvt_dense).astype(np.float32),
        'spatial_vs_dense': (tvt_fs['tvtF_ANCC']-tvt_dense).astype(np.float32),
        'beam_vs_spatial': (bpaths['cons']-tvt_fs['tvtF_ANCC']).astype(np.float32),
        'sc_vs_beam': (sc_ens-bpaths['cons']).astype(np.float32),
        'beam_vs_pf': (beam_ref-pf_a).astype(np.float32),
        'dtw_vs_pf': (dtw_ens_ev-pf_a).astype(np.float32),
        'dtw_vs_beam': (dtw_ens_ev-beam_ref).astype(np.float32),
        # DTW multi-scale
        'dtw_ens_d': (dtw_ens_ev-last_tvt).astype(np.float32),
        **{f'dtw_r{r}_d': (dtw_per_r[r]-last_tvt).astype(np.float32) for r in DTW_RADII},
        **{f'dtw_slope_r{r}': dtw_slope_ev[r] for r in DTW_RADII},
        'dtw_slope_mean': np.stack([dtw_slope_ev[r] for r in DTW_RADII],1).mean(1).astype(np.float32),
        **{f'dtw_cost_r{r}': sc(dtw_costs[r]) for r in DTW_RADII},
        'dtw_cost_min': sc(dtw_cost_min), 'dtw_cost_range': sc(dtw_cost_range),
        # DTW stochastic
        'dtw_stoch_mean_d': (dtw_mean_ev-last_tvt).astype(np.float32),
        'dtw_stoch_std': dtw_std_ev, 'dtw_stoch_cv': dtw_cv_ev,
        # Derivative DTW (aligns dGR/dMD — formation boundaries)
        'dtw_deriv_ens_d': (dtw_deriv_ens_ev-last_tvt).astype(np.float32),
        **{f'dtw_deriv_r{r}_d': (dtw_deriv_per_r[r]-last_tvt).astype(np.float32) for r in DTW_RADII},
        **{f'dtw_deriv_cost_r{r}': sc(dtw_deriv_costs[r]) for r in DTW_RADII},
        'dtw_deriv_cost_min': sc(dtw_deriv_cost_min),
        'dtw_deriv_vs_dtw': (dtw_deriv_ens_ev-dtw_ens_ev).astype(np.float32),
        'dtw_deriv_vs_pf': (dtw_deriv_ens_ev-pf_a).astype(np.float32),
        # Anchor-constrained DTW (US11391856: soft Gaussian constraints at known TVT positions)
        'dtw_anc_ens_d': (dtw_anc_ens_ev-last_tvt).astype(np.float32),
        **{f'dtw_anc_r{r}_d': (dtw_anc_per_r[r]-last_tvt).astype(np.float32) for r in DTW_RADII},
        **{f'dtw_anc_cost_r{r}': sc(dtw_anc_costs[r]) for r in DTW_RADII},
        'dtw_anc_cost_min': sc(dtw_anc_cost_min),
        'dtw_anc_vs_free': (dtw_anc_ens_ev-dtw_ens_ev).astype(np.float32),  # how much anchors changed DTW
        'dtw_anc_vs_pf': (dtw_anc_ens_ev-pf_a).astype(np.float32),
        # Affine calibration + stats
        'cal_a': sc(a_cal), 'cal_b': sc(b_cal), 'pfx_rmse': sc(pfx_rmse),
        # Slope features
        'slp_all': sc(slp_all), 'slp_50': sc(slp_50), 'slp_z': sc(slp_z),
        'slp_b_d_all': (slp_b_all-last_tvt).astype(np.float32),
        'slp_b_d_50': (slp_b_50-last_tvt).astype(np.float32),
        # Known section stats
        'known_len': sc(len(kn)), 'eval_len': sc(nh),
        'ktvt_range': sc(float(np.ptp(ktvt))), 'ktvt_std': sc(float(ktvt.std())),
        # Geometry
        'md_since': md_since, 'frac': frac, 'frac2': frac**2, 'sqrt_frac': np.sqrt(frac),
        'z': z_ev,
        'dx': (ev['X']-float(lk['X'])).to_numpy(np.float32),
        'dy': (ev['Y']-float(lk['Y'])).to_numpy(np.float32),
        'dz': (z_ev-float(lk['Z'])).astype(np.float32),
        'dxy': np.sqrt((ev['X']-float(lk['X']))**2+(ev['Y']-float(lk['Y']))**2).to_numpy(np.float32),
        'dzdmd': dzdmd, 'dxdmd': dxdmd, 'dydmd': dydmd,
        # GR features
        'gr': hgr_ev, 'gr_d1': gr_d1, 'gr_d2': gr_d2, 'gr_env': gr_env, 'gr_nrg': gr_nrg,
        **{f'grm{w}': gr_s.rolling(w,center=True,min_periods=1).mean().values[ev_start:ev_start+nh].astype(np.float32) for w in [5,21,51,101]},
        **{f'grs{w}': gr_s.rolling(w,center=True,min_periods=1).std().fillna(0).values[ev_start:ev_start+nh].astype(np.float32) for w in [5,21,51]},
        **{f'glag{lag}': gr_s.shift(lag).bfill().values[ev_start:ev_start+nh].astype(np.float32) for lag in [1,5,15,30]},
        **{f'glead{lag}': gr_s.shift(-lag).ffill().values[ev_start:ev_start+nh].astype(np.float32) for lag in [1,5,15,30]},
        # ── Wavelet (DWT) features on GR signal ──
        **_wavelet_features(gr_full, ev_start, nh),
        # GR residuals at offsets
        **{f'tda{int(o)}':   hgr_ev-np.float32(tw_interp(last_tvt+o)) for o in ANCH_OFFS},
        **{f'tdbc{int(o)}':  hgr_ev-tw_interp(beam_ref+o).astype(np.float32) for o in BEAM_OFFS},
        **{f'tdsc{int(o)}':  hgr_ev-tw_interp(sc_ens+o).astype(np.float32) for o in SC_OFFS},
        **{f'tdpf{int(o)}':  hgr_ev-tw_interp(pf_a+o).astype(np.float32) for o in PF_OFFS},
        **{f'tddtw{int(o)}': hgr_ev-tw_interp(dtw_ens_ev+o).astype(np.float32) for o in DTW_OFFS},
        # Derivative DTW offset scan
        **{f'tddtwd{int(o)}': hgr_ev-tw_interp(dtw_deriv_ens_ev+o).astype(np.float32) for o in DTW_OFFS},
        **{f'tddtwanc{int(o)}': hgr_ev-tw_interp(dtw_anc_ens_ev+o).astype(np.float32) for o in DTW_OFFS},
        # Wavelet-typewell correlation (similarity of GR power spectrum at pred TVT)
        'wav_corr_pf':   _wavelet_corr_features(hgr_ev, tw_tvt, tw_gr, pf_a[:nh]),
        'wav_corr_beam': _wavelet_corr_features(hgr_ev, tw_tvt, tw_gr, beam_ref[:nh]),
        'wav_corr_dtw':  _wavelet_corr_features(hgr_ev, tw_tvt, tw_gr, dtw_ens_ev),
    }

    result=pd.DataFrame(feats)
    if is_train:
        if 'TVT' not in ev.columns or ev['TVT'].isna().all(): return None
        result['target']=(ev['TVT'].to_numpy(np.float32)-np.float32(last_tvt))
    return result

# ─────────────────────────── Main ─────────────────────────────

t0 = time.time()

import json as _json
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import optuna; optuna.logging.set_verbosity(optuna.logging.WARNING)

# Paths defined at top of file (COMP_DIR, MODELS_DS, MODEL_DIR, TCN_DS)

# ── Load trained models ───────────────────────────────────────────────────────
print('Loading trained models...', flush=True)
feat_cols  = _json.loads(open(MODEL_DIR / 'feat_cols.json').read())
blend_info = _json.loads(open(MODEL_DIR / 'blend_weights.json').read())
weights    = blend_info['weights']
n_lgbm     = blend_info['n_lgbm']
n_xgb      = blend_info['n_xgb']
n_cb       = blend_info['n_cb']
CNN_WEIGHT = blend_info.get('cnn_weight', 0.0)
N_FOLDS    = 5
print(f'Blend: {n_lgbm} LGBM + {n_xgb} XGB + {n_cb} CB, CNN weight={CNN_WEIGHT:.2f}', flush=True)

lgb_models = [[lgb.Booster(model_file=str(MODEL_DIR / f'lgbm_{mi}_fold{f}.txt'))
               for f in range(N_FOLDS)] for mi in range(n_lgbm)]
xgb_models = []
for f in range(N_FOLDS):
    m = xgb.XGBRegressor(); m.load_model(str(MODEL_DIR / f'xgb_fold{f}.json'))
    xgb_models.append(m)
cb_models = []
for mi in range(n_cb):
    fold_ms = []
    for f in range(N_FOLDS):
        m = CatBoostRegressor(verbose=0); m.load_model(str(MODEL_DIR / f'cb_{mi}_fold{f}.cbm'))
        fold_ms.append(m)
    cb_models.append(fold_ms)

oof_meta = pd.read_parquet(MODELS_DS / 'oof_meta.parquet')
print(f'Models loaded: {n_lgbm} LGBM + {n_xgb} XGB + {n_cb} CatBoost, {N_FOLDS} folds each', flush=True)

# ── Build spatial indices then generate test features on-the-fly ─────────────
print('Building spatial indices...', flush=True)
t_idx = time.time()
train_wells = list_wells(TRAIN_DIR)
test_wells  = list_wells(TEST_DIR)
_FI = FormationPlaneKNN(train_wells, TRAIN_DIR)
_DI = DenseANCCImputer(train_wells, TRAIN_DIR)
print(f'Indices ready in {time.time()-t_idx:.1f}s', flush=True)

print('Building test features on-the-fly...', flush=True)
t2 = time.time()
test_parts = [build_well(wid, TEST_DIR, is_train=False) for wid in test_wells]
test_df = pd.concat([p for p in test_parts if p is not None], ignore_index=True)
print(f'Test features: {test_df.shape}  ({time.time()-t2:.1f}s)', flush=True)

feat_cols = [c for c in feat_cols if c in test_df.columns]
print(f'Loaded in {time.time()-t0:.1f}s', flush=True)
print(f'Test: {test_df.shape}, Features: {len(feat_cols)}', flush=True)

# ── Predict with loaded models ────────────────────────────────────────────────
X_test = np.nan_to_num(test_df[feat_cols].values.astype(np.float32), nan=0., posinf=0., neginf=0.)

all_test_preds = []
for mi, fold_ms in enumerate(lgb_models):
    p = np.mean([m.predict(X_test) for m in fold_ms], axis=0).astype(np.float32)
    all_test_preds.append(p)
xgb_test = np.mean([m.predict(X_test) for m in xgb_models], axis=0).astype(np.float32)
all_test_preds.append(xgb_test)
for mi, fold_ms in enumerate(cb_models):
    p = np.mean([m.predict(X_test) for m in fold_ms], axis=0).astype(np.float32)
    all_test_preds.append(p)

test_stack = np.stack(all_test_preds, axis=1)
ens_test_gbm = (test_stack * np.array(weights)).sum(1).astype(np.float32)

# ── CNN inference ─────────────────────────────────────────────────────────────
CNN_N_FOLDS = 5
WINDOW = 40

cnn_test = np.zeros(len(test_df), np.float32)
try:
    import torch
    import torch.nn as nn

    class WellCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(3, 32, 7, padding=3), nn.BatchNorm1d(32), nn.GELU(),
                nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.GELU(),
                nn.Conv1d(64, 64, 3, padding=1), nn.BatchNorm1d(64), nn.GELU(),
            )
            self.head = nn.Sequential(
                nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1)
            )
        def forward(self, x):
            h = self.net(x)
            c = h[:, :, WINDOW]
            return self.head(c).squeeze(-1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build test windows from raw test CSVs
    test_wins = []
    test_win_idx = []
    for wid in test_df['well'].unique():
        csv_path = TEST_DIR / f'{wid}__horizontal_well.csv'
        if not csv_path.exists():
            continue
        raw = pd.read_csv(csv_path)
        unknown_mask = raw['TVT_input'].isna()
        if 'ANCC' not in raw.columns: raw['ANCC'] = 0.0
        sigs = raw[['GR', 'ANCC', 'Z']].values.astype(np.float32)
        for ch in range(3):
            mu, sd = sigs[:, ch].mean(), sigs[:, ch].std() + 1e-6
            sigs[:, ch] = (sigs[:, ch] - mu) / sd
        padded = np.pad(sigs, ((WINDOW, WINDOW), (0, 0)), mode='edge')
        wdf = test_df[test_df['well'] == wid]
        unk_raw_indices = np.where(unknown_mask.values)[0]
        if len(unk_raw_indices) != len(wdf):
            unk_raw_indices = unk_raw_indices[:len(wdf)]
        for i_wdf, i_raw in zip(wdf.index, unk_raw_indices):
            win = padded[i_raw: i_raw + 2*WINDOW + 1, :].T
            test_wins.append(win)
            test_win_idx.append(i_wdf)

    cnn_fold_preds = []
    cnn_loaded = False
    for fold in range(CNN_N_FOLDS):
        mp = MODEL_DIR / f'cnn_fold{fold}.pt'
        if not mp.exists():
            continue
        m = WellCNN().to(device)
        m.load_state_dict(torch.load(str(mp), map_location=device))
        m.eval()
        if test_wins:
            X_t = torch.tensor(np.stack(test_wins), dtype=torch.float32).to(device)
            with torch.no_grad():
                p = m(X_t).cpu().numpy().astype(np.float32)
        else:
            p = np.zeros(len(test_win_idx), np.float32)
        cnn_fold_preds.append(p)
        cnn_loaded = True

    if cnn_loaded and test_win_idx:
        cnn_win_pred = np.stack(cnn_fold_preds, axis=0).mean(0)
        for pos, idx in enumerate(test_win_idx):
            cnn_test[test_df.index.get_loc(idx)] = cnn_win_pred[pos]
        print(f'CNN test predictions: mean={cnn_test.mean():.4f}', flush=True)
    else:
        print('CNN models not found or no test windows, skipping CNN blend', flush=True)

except Exception as e:
    print(f'CNN inference failed: {e}', flush=True)
    cnn_test = None

if CNN_WEIGHT > 0 and cnn_test is not None and cnn_test.any():
    ens_test = ((1 - CNN_WEIGHT) * ens_test_gbm + CNN_WEIGHT * cnn_test).astype(np.float32)
    print(f'GBM+CNN blend applied at CNN weight={CNN_WEIGHT:.2f}', flush=True)
else:
    ens_test = ens_test_gbm.copy()
    print('CNN weight=0 or unavailable, using GBM-only predictions', flush=True)

# ── Optuna PP (fit on OOF, apply to test) ────────────────────────────────────
y_np     = oof_meta['target'].values.astype(np.float32)
# Support both GPU-kernel column names and local-training column names
ens_oof  = oof_meta.get('ens_oof',  oof_meta.get('oof_pred')).values.astype(np.float32)
pf_oof   = oof_meta.get('pf_ancc_d', oof_meta.get('pf_ancc_delta')).values.astype(np.float32)
form_oof = oof_meta.get('form_d',    oof_meta.get('pf_ancc_delta')).values.astype(np.float32)
last_tvt = oof_meta['last_known_tvt'].values
md_since = oof_meta['md_since'].values
ytrue_abs = y_np + last_tvt

pf_test   = test_df['pf_ancc'].values - test_df['last_known_tvt'].values
form_cols_t = [c for c in test_df.columns if c.startswith('tvtF_') and not c.startswith('tvtFw') and not c.startswith('tvtF5')]
form_test = test_df[form_cols_t].mean(1).values - test_df['last_known_tvt'].values if form_cols_t else pf_test
last_tvt_t = test_df['last_known_tvt'].values
md_since_t = test_df['md_since'].values

def objective(trial):
    alpha  = trial.suggest_float('alpha', 0.3, 1.5)
    tau    = trial.suggest_float('tau', 50, 1000)
    w_pf   = trial.suggest_float('w_pf', 0.0, 0.4)
    w_form = trial.suggest_float('w_form', 0.0, 0.3)
    d_blend = (1-w_pf-w_form)*ens_oof + w_pf*pf_oof + w_form*form_oof
    tau_fac = 1. - np.exp(-np.maximum(md_since, 0.) / tau)
    d_pp = (1.-tau_fac)*pf_oof + tau_fac*alpha*d_blend
    return float(np.sqrt(np.mean((last_tvt + d_pp - ytrue_abs)**2)))

study = optuna.create_study(direction='minimize',
                            sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=50))
study.optimize(objective, n_trials=500, n_jobs=-1)
bp = study.best_params
print(f'PP RMSE={study.best_value:.4f} params={bp}', flush=True)

w_mdl_t = 1. - bp['w_pf'] - bp['w_form']
d_blend_t = w_mdl_t*ens_test + bp['w_pf']*pf_test + bp['w_form']*form_test
tau_fac_t = 1. - np.exp(-np.maximum(md_since_t, 0.) / bp['tau'])
test_pp = (1.-tau_fac_t)*pf_test + tau_fac_t*bp['alpha']*d_blend_t
test_df['pred'] = last_tvt_t + test_pp

for _, grp in test_df.groupby('well', sort=False):
    v = grp['pred'].values
    wl = min(17, len(v)); wl -= (wl%2==0)  # SG=17 restored (61 hurt LB)
    if wl >= 5: v = savgol_filter(v, wl, 3)
    test_df.loc[grp.index, 'pred'] = v

# ── TCN v3 blend (OOF 7.57 — non-causal+RevIN+sliding-windows+anchor-channels) ─
# Find TCN models zip — check for the FILE not just the directory
# (Kaggle creates empty placeholder dirs even when dataset content isn't mounted)
_tcn_cfg_candidates = [
    TCN_DS / 'kaggle_tcn_dataset' / 'tcn_config.json',
    TCN_DS / 'tcn_config.json',
]
_found_cfg = next((p for p in _tcn_cfg_candidates if p.exists()), None)
print(f'TCN config search: {[(str(p), p.exists()) for p in _tcn_cfg_candidates]}', flush=True)
TCN_DIR = _found_cfg.parent if _found_cfg else TCN_DS
tcn_pred = np.zeros(len(test_df), np.float32)
tcn_available = False
try:
    import torch, torch.nn as nn, torch.nn.functional as F, json as _tcn_json
    _tcn_tmp = TCN_DIR
    if _found_cfg is None:
        raise FileNotFoundError('tcn_config.json not found — add rogii-tcn-v4-models2 dataset to kernel inputs')
    if (_tcn_tmp / 'tcn_config.json').exists():
        _cfg = _tcn_json.load(open(_tcn_tmp / 'tcn_config.json'))
        C_IN_T=_cfg.get('c_in',7); C_HID_T=_cfg.get('c_hid',64)
        KRN_T=_cfg.get('kernel',7); DIL_T=_cfg.get('dilations',[1,2,4,8,16,32,64,128])

        class _SC(nn.Module):
            def __init__(self,c,k,d):
                super().__init__(); self.pad=((k-1)*d)//2
                self.conv=nn.Conv1d(c,c,k,dilation=d,padding=self.pad)
                self.norm=nn.LayerNorm(c); self.act=nn.GELU()
            def forward(self,x):
                h=self.conv(x)
                if h.shape[-1]!=x.shape[-1]: h=h[...,:x.shape[-1]]
                return self.act(self.norm(h.transpose(1,2)).transpose(1,2))

        class _TB(nn.Module):
            def __init__(self,c,k,d):
                super().__init__()
                self.conv_f=_SC(c,k,d); self.conv_g=_SC(c,k,d)
                self.res=nn.Conv1d(c,c,1); self.skip=nn.Conv1d(c,c,1)
            def forward(self,x):
                h=torch.tanh(self.conv_f(x))*torch.sigmoid(self.conv_g(x))
                return self.res(h)+x, self.skip(h)

        class _TM(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj=nn.Conv1d(C_IN_T,C_HID_T,1)
                self.blocks=nn.ModuleList([_TB(C_HID_T,KRN_T,d) for d in DIL_T])
                self.head=nn.Sequential(nn.LayerNorm(C_HID_T),nn.Linear(C_HID_T,32),nn.GELU(),nn.Linear(32,1))
            def forward(self,x):
                h=self.proj(x); s=torch.zeros_like(h)
                for b in self.blocks: h,sk=b(h); s=s+sk
                return self.head(F.gelu(s).transpose(1,2)).squeeze(-1)

        print(f'TCN config loaded: {_cfg}', flush=True)
        _dev=torch.device('cpu')
        _folds=[]; _nf=_cfg.get('n_folds',5)
        for _f in range(_nf):
            # Try .pkl first (v4 dataset), fall back to .pt (v3)
            _mp=_tcn_tmp/f'tcn_fold{_f}_best.pkl'
            if not _mp.exists(): _mp=_tcn_tmp/f'tcn_fold{_f}.pkl'
            if not _mp.exists(): _mp=_tcn_tmp/f'tcn_fold{_f}_best.pt'
            if not _mp.exists(): _mp=_tcn_tmp/f'tcn_fold{_f}.pt'
            if _mp.exists():
                _m=_TM(); _m.load_state_dict(torch.load(str(_mp),map_location=_dev)); _m.eval(); _folds.append(_m)
        print(f'TCN: {len(_folds)} fold models loaded', flush=True)

        for wid in test_df['well'].unique():
            try:
                raw=pd.read_csv(TEST_DIR/f'{wid}__horizontal_well.csv')
                ev_mask=raw['TVT_input'].isna().values
                if ev_mask.sum()==0: continue
                kn=raw[~ev_mask]; last_tvt_w=float(kn['TVT_input'].iloc[-1])
                sig=raw[['GR','Z']].interpolate(limit_direction='both').fillna(0.).values.astype(np.float32)
                for _c in range(2):
                    mu,sd=sig[:,_c].mean(),sig[:,_c].std()+1e-6; sig[:,_c]=(sig[:,_c]-mu)/sd
                tvt_in=raw['TVT_input'].values.astype(np.float32)
                tvt_f=np.where(np.isnan(tvt_in),0.,(tvt_in-np.nanmean(tvt_in))/(np.nanstd(tvt_in)+1e-6))
                tvt_m=(~np.isnan(tvt_in)).astype(np.float32)
                L=len(raw); md=np.arange(L,dtype=np.float32)/max(L-1,1)
                X=np.stack([sig[:,0],sig[:,1],tvt_f,tvt_m,np.sin(md*2*np.pi),np.cos(md*2*np.pi)],axis=0)
                X_t=torch.from_numpy(X).unsqueeze(0)
                wgrp=test_df[test_df['well']==wid]
                # RevIN denorm: use GBM predictions as proxy for eval-position delta stats.
                # Training RevIN used mean/std of (TVT - last_tvt) at EVAL positions (positive,
                # increasing). Using anchor deltas was wrong — all anchors are BEFORE last_tvt_w
                # so their deltas are negative, causing massive negative shift.
                gbm_delta_ev=wgrp['pred'].values-last_tvt_w  # positive for eval positions
                w_mean=float(np.mean(gbm_delta_ev)); w_std=float(np.std(gbm_delta_ev))+1e-6
                print(f'  well {wid}: {len(kn)} anchors, w_mean={w_mean:.1f} w_std={w_std:.1f}', flush=True)
                fp=[]
                with torch.no_grad():
                    for _m in _folds:
                        out=_m(X_t).numpy()[0]  # (L,) normalized delta
                        out_ev=out[ev_mask][:len(wgrp)]
                        fp.append(out_ev*w_std+w_mean)  # denorm
                avg=np.mean(fp,axis=0)
                tcn_pred[wgrp.index]=avg.astype(np.float32); tcn_available=True
            except Exception as _we:
                import traceback as _wetb
                print(f'TCN well {wid} failed: {_we}\n{_wetb.format_exc()[-800:]}', flush=True)

        nz=(tcn_pred!=0).sum()
        print(f'TCN: {nz}/{len(test_df)} rows, range=[{tcn_pred.min():.2f},{tcn_pred.max():.2f}]', flush=True)
except Exception as _e:
    import traceback as _tb
    print(f'TCN failed: {_e}', flush=True)
    print(_tb.format_exc()[-500:], flush=True)

if tcn_available and (tcn_pred!=0).mean()>0.5:
    TCN_W=0.40  # OOF sweep: 40% optimal (9.297) vs 50% (9.432); TCN adds diversity not accuracy
    gbm_abs=test_df['pred'].values  # GBM pred is already absolute TVT
    tcn_abs=test_df['last_known_tvt'].values+tcn_pred  # TCN delta → absolute
    test_df['pred']=(1-TCN_W)*gbm_abs+TCN_W*tcn_abs
    print(f'{TCN_W:.0%} TCN + {1-TCN_W:.0%} GBM blend applied', flush=True)
else:
    print('TCN unavailable, GBM-only', flush=True)

sample = pd.read_csv(COMP_DIR / 'sample_submission.csv')
id2pred = dict(zip(test_df['id'], test_df['pred']))
sample['tvt'] = sample['id'].map(id2pred).fillna(test_df['pred'].median())
sample[['id','tvt']].to_csv('/kaggle/working/submission.csv', index=False)
print(f'Submission: {len(sample)} rows, TVT=[{sample["tvt"].min():.2f},{sample["tvt"].max():.2f}]', flush=True)
print(f'Total time: {time.time()-t0:.1f}s', flush=True)
print(f'Done: submission.csv written', flush=True)

