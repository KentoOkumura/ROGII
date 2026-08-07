# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # exp510 exp413 exact public pre-override hedge inference
#
# ## Contents
#
# 1. Imports and immutable source contract
# 2. Runtime, SHA, and fail-closed input helpers
# 3. Public-source physics and feature helpers
# 4. Setup and input contract
# 5. Exact projected-SP45 component generation
# 6. Saved Pipeline-B inference
# 7. Dynamic hidden-safe exp413 generation and fixed hedge
# 8. Metrics, diagnostics, and generated artifacts
#
# This candidate stops at the archived source's visible pre-override boundary.
# The projected-SP45 branch must cover every sample ID, so the source's
# Pipeline-A tabular fallback and all of its training code are absent. Pipeline
# B loads exactly three SHA-pinned boosters; missing artifacts fail closed.

# %% [markdown]
# ## 1. Imports and immutable source contract

# %%
import gzip
import hashlib
import json
import multiprocessing
import os
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.signal import savgol_filter
from scipy.spatial import cKDTree

from exp510_exp413_hidden_safe_runtime import (
    PARENT_SOURCE_SHA256 as EXP413_PARENT_SOURCE_SHA256,
    generate_dynamic_exp413_prediction,
)

warnings.filterwarnings("ignore")

BASE_SEED = 42
SOURCE_KERNEL_ID = "degnonguidi/public-score-rogii-lb-7-159"
SOURCE_SHA256 = "4d0712983788dc7d9b97fdb8e5dc7c30b6d3634a9c64597d84d21da28e9623eb"
MODEL_DATASET = "fleongg/rogii-claude-models-pub"
MODEL_DATASET_VERSION = 1
MODEL_SHA256 = {
    "features.json": "ea9042f88cb3d8716b83e40c5c5ecb39f8bc8fcfeb52edb40d1871cd99496308",
    "lgb0.pkl": "a6451b3c42aeace6778e952b088287654946dca5412b818990d3f6b397e501e1",
    "lgb1.pkl": "4d61ab162af864bd3cfe37bde4421299746f28147faa3239e1ad14f15453f547",
    "lgb2.pkl": "1ee24121ecf455d904f3433bba49857d076fc33ca0b6b7a71ff9d538b3b8acf5",
}
EXPECTED_ZERO_FILLED_FEATURES = {
    "beam_stiff_d",
    "beam_vcons_d",
    "beam_vloose_d",
    *{f"tda{offset}" for offset in (-80, -40, -20, -10, 10, 20, 40, 80)},
    *{f"tdbc{offset}" for offset in (-40, -20, -10, -3, 3, 10, 20, 40)},
    *{f"tdsc{offset}" for offset in (-30, -15, -8, -4, -2, 2, 4, 8, 15, 30)},
    *{f"tdpf{offset}" for offset in (-30, -15, -8, -4, -2, 2, 4, 8, 15, 30)},
}
EXP413_VISIBLE_REFERENCE_CONTENT_SHA256 = (
    "875a1334ae3c90f841414f8f98d8877fb06234e17e0fd0b8d46385170a584dc4"
)
SP45_WEIGHT = np.float64(0.55)
PIPELINE_B_WEIGHT = np.float64(0.45)
EXP413_WEIGHT = np.float64(0.90)
PUBLIC_WEIGHT = np.float64(0.10)
FORMATIONS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]


# %% [markdown]
# ## 2. Runtime, SHA, and fail-closed input helpers

# %%
def find_data_dir():
    candidates = [
        Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction"),
        Path("/kaggle/input/rogii-wellbore-geology-prediction"),
        Path(os.environ.get("ROGII_DATA", ".")),
    ]
    for candidate in candidates:
        if (
            (candidate / "sample_submission.csv").is_file()
            and (candidate / "train").is_dir()
            and (candidate / "test").is_dir()
        ):
            return candidate.resolve()
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for sample_path in sorted(kaggle_input.rglob("sample_submission.csv")):
            candidate = sample_path.parent
            if (candidate / "train").is_dir() and (candidate / "test").is_dir():
                return candidate.resolve()
    raise FileNotFoundError("competition data root with sample/train/test was not found")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_gzip_content(path):
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(split, family, well, seed_index, base_seed=BASE_SEED):
    token = f"{base_seed}|{split}|{family}|{well}|{seed_index}".encode()
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "little") % (2**32)


def dataframe_content_sha(frame, columns):
    payload = frame.loc[:, columns].to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.17g",
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def schema_sha(frame):
    schema = [(str(col), str(frame[col].dtype)) for col in frame.columns]
    return hashlib.sha256(json.dumps(schema, separators=(",", ":")).encode()).hexdigest()


def find_exact_model_dir(search_roots=None):
    roots = [Path(path) for path in (search_roots or ["/kaggle/input", "."])]
    matches = []
    for root in roots:
        if not root.exists():
            continue
        for feature_path in root.rglob("features.json"):
            model_dir = feature_path.parent
            if not all((model_dir / name).is_file() for name in MODEL_SHA256):
                continue
            observed = {name: sha256_file(model_dir / name) for name in MODEL_SHA256}
            if observed == MODEL_SHA256:
                matches.append(model_dir.resolve())
    matches = sorted(set(matches))
    if len(matches) != 1:
        raise RuntimeError(f"expected one SHA-matched Pipeline-B model dir, got {matches}")
    return matches[0]


class CFG:
    DATA = None
    OUT = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
    n_jobs = min(8, multiprocessing.cpu_count())


# %% [markdown]
# ## 3. Public-source physics and feature helpers

# %%
def load_well(wid, split="train", data_dir=None):
    base = (data_dir or CFG.DATA) / split
    hw = pd.read_csv(base / f"{wid}__horizontal_well.csv")
    tw = pd.read_csv(base / f"{wid}__typewell.csv").sort_values("TVT")
    return hw, tw


def rmse(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() == 0:
        return float("inf")
    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2)))


# ── single-seed likelihood particle filter ─────────────────────────────────
def run_particle_filter(hw, tw, n_particles=500, seed=42):
    tw_s = tw.sort_values("TVT")
    tw_tvt = tw_s["TVT"].values.astype(float)
    tw_gr = tw_s["GR"].fillna(tw_s["GR"].mean()).values.astype(float)

    kn = hw[hw["TVT_input"].notna()]
    ev = hw[hw["TVT_input"].isna()]
    if len(ev) == 0:
        return hw["TVT_input"].values.astype(float).copy(), 0.0

    last = kn.iloc[-1]
    last_tvt = float(last["TVT_input"])
    last_Z = float(last["Z"])
    last_MD = float(last["MD"])

    tw_at_k = np.interp(kn["TVT_input"].values, tw_tvt, tw_gr)
    gs = float(np.clip(np.nanstd(kn["GR"].fillna(0).values - tw_at_k), 10., 60.))

    tail = kn.tail(30)
    dt = np.diff(tail["TVT_input"].values)
    dz = np.diff(tail["Z"].values)
    dm = np.diff(tail["MD"].values)
    m = dm > 0
    ir = float(np.median((dt + dz)[m] / dm[m])) if m.sum() >= 3 else 0.0

    N = n_particles
    rng = np.random.default_rng(seed)
    ls = last_tvt + last_Z
    pos = ls + 4.5 * rng.standard_normal(N)
    rate = ir + 0.01 * rng.standard_normal(N)
    w = np.ones(N) / N

    MOM, VN, PN, RP, RR, RESAMP = 0.998, 0.002, 0.005, 0.1, 0.001, 0.5

    md_v = ev["MD"].values.astype(float)
    z_v = ev["Z"].values.astype(float)
    gr_interp = hw["GR"].interpolate(limit_direction="both").fillna(tw_gr.mean())
    gr_v = gr_interp.values.astype(float)[ev.index]

    out_vals = hw["TVT_input"].values.astype(float).copy()
    res = np.empty(len(ev))
    prev_MD = last_MD
    log_lik = 0.0

    for i in range(len(ev)):
        dm_step = max(md_v[i] - prev_MD, 1.0)
        rate = MOM * rate + VN * rng.standard_normal(N)
        pos = pos + rate * dm_step + PN * rng.standard_normal(N)
        tvt_p = pos - z_v[i]
        tvt_p = np.clip(tvt_p, tw_tvt[0] - 100, tw_tvt[-1] + 100)
        pos = tvt_p + z_v[i]

        eg = np.interp(tvt_p, tw_tvt, tw_gr)
        d = (gr_v[i] - eg) / gs
        lk = np.exp(-0.5 * np.minimum(d ** 2, 600.))
        lk = np.maximum(lk, 1e-300)
        avg_lk = float((w * lk).sum())
        log_lik += np.log(max(avg_lk, 1e-300))
        w = w * lk
        ws = w.sum()
        w = w / ws if ws > 0 else np.ones(N) / N

        n_eff = 1.0 / (w ** 2).sum()
        if n_eff < RESAMP * N:
            cum = np.cumsum(w)
            u0 = rng.uniform(0, 1.0 / N)
            idx = np.clip(np.searchsorted(cum, u0 + np.arange(N) / N), 0, N - 1)
            pos = pos[idx] + RP * rng.standard_normal(N)
            rate = rate[idx] + RR * rng.standard_normal(N)
            w = np.ones(N) / N

        res[i] = float(np.dot(w, pos - z_v[i]))
        prev_MD = md_v[i]

    out_vals[list(ev.index)] = res
    return out_vals, log_lik


def run_pf_lik_ensemble(
    hw,
    tw,
    well,
    split,
    family,
    n_particles=500,
    n_seeds=128,
    scale=5.0,
):
    preds, liks = [], []
    for seed_index in range(n_seeds):
        seed = stable_seed(split, family, well, seed_index)
        p, ll = run_particle_filter(hw, tw, n_particles=n_particles, seed=seed)
        preds.append(p)
        liks.append(ll)
    liks = np.array(liks)
    liks_n = liks - liks.max()
    weights = np.exp(liks_n / scale)
    weights /= weights.sum()
    return (weights[:, None] * np.stack(preds, 0)).sum(0)


SELECTOR_SCALES = (3.0, 5.0, 8.0, 12.0)


def run_pf_lik_ensemble_scales(
    hw,
    tw,
    well,
    split,
    family,
    scales=SELECTOR_SCALES,
    n_particles=500,
    n_seeds=128,
):
    out = {}
    preds, liks = [], []
    for seed_index in range(n_seeds):
        seed = stable_seed(split, family, well, seed_index)
        p, ll = run_particle_filter(hw, tw, n_particles=n_particles, seed=seed)
        preds.append(p)
        liks.append(ll)
    preds = np.stack(preds, 0)
    liks = np.array(liks)
    liks_n = liks - liks.max()
    for scale in scales:
        weights = np.exp(liks_n / scale)
        weights /= weights.sum()
        out[f"scale_{scale:g}"] = (weights[:, None] * preds).sum(0)
    out["pf_mean"] = preds.mean(0)
    return out



# %%
# ── global GR-alignment beam search ────────────────────────────────────────
BEAM_CONFIGS = [
    (10, 20.0, 144.0, 2), (10, 8.0, 64.0, 2), (8, 35.0, 220.0, 1),
    (10, 14.0, 90.0, 5), (20, 4.0, 36.0, 3), (12, 12.0, 100.0, 3),
    (15, 25.0, 180.0, 2), (20, 30.0, 200.0, 2), (15, 10.0, 80.0, 4),
    (25, 6.0, 50.0, 3), (10, 40.0, 300.0, 1), (12, 18.0, 120.0, 5),
    (30, 8.0, 70.0, 2), (10, 50.0, 400.0, 0),
]

# Named subset used for per-well feature deltas (tags chosen to mirror the
# 'cons' / 'sm5' references used by the multi-scale-NCC blend below).
BEAMS = [
    (10, 20.0, 144.0, 2, "cons"),
    (20, 4.0, 36.0, 3, "sm5"),
    (8, 35.0, 220.0, 1, "wide"),
    (15, 10.0, 80.0, 4, "tight"),
    (25, 6.0, 50.0, 3, "fine"),
    (10, 50.0, 400.0, 0, "loose"),
    (12, 18.0, 120.0, 5, "mid"),
]


def beam_search(hgr, tw_tvt, tw_gr, last_tvt, bs=10, mc=20.0, es=144.0, r=2):
    # Beam search ±r: TVT can move down or up by up to r steps per MD sample,
    # constrained by max-change (mc) and excursion-span (es) GR-alignment cost.
    n = len(hgr)
    if n == 0:
        return np.array([], dtype=np.float32)
    grid = np.arange(tw_tvt.min() - 50, tw_tvt.max() + 50, 0.5)
    tw_interp = np.interp(grid, tw_tvt, tw_gr)

    beams = [(last_tvt, 0.0)] * bs
    out = np.empty(n, dtype=np.float32)
    cur = last_tvt
    for i in range(n):
        candidates = []
        step = mc / max(bs, 1)
        for k in range(-r, r + 1):
            cand_tvt = cur + k * step
            idx = int(np.clip(np.searchsorted(grid, cand_tvt), 0, len(grid) - 1))
            cost = abs(hgr[i] - tw_interp[idx])
            candidates.append((cost, cand_tvt))
        candidates.sort(key=lambda x: x[0])
        cur = candidates[0][1]
        cur = float(np.clip(cur, tw_tvt.min() - es, tw_tvt.max() + es))
        out[i] = cur
    return out


def run_beam_ensemble(hw, tw):
    tw_s = tw.sort_values("TVT")
    tw_tvt = tw_s["TVT"].values.astype(float)
    tw_gr = tw_s["GR"].fillna(tw_s["GR"].mean()).values.astype(float)
    kn = hw[hw["TVT_input"].notna()]
    ev = hw[hw["TVT_input"].isna()]
    if len(ev) == 0:
        return hw["TVT_input"].values.astype(float).copy()
    last_tvt = float(kn["TVT_input"].iloc[-1])
    gr_full = hw["GR"].astype(float).interpolate(limit_direction="both").fillna(float(np.nanmean(tw_gr)))
    hgr = gr_full.iloc[ev.index[0]:].to_numpy(np.float32)

    paths = [beam_search(hgr, tw_tvt, tw_gr, last_tvt, bs, mc, es, r) for (bs, mc, es, r) in BEAM_CONFIGS]
    ens = np.mean(np.stack(paths, 0), axis=0)
    out_vals = hw["TVT_input"].values.astype(float).copy()
    out_vals[list(ev.index)] = ens
    return out_vals



# %%
# ── offset-well spatial priors: plane-KNN through formation tops + dense ANCC surface ──
class FormationPlaneKNN:
    # Local plane through each formation top, fit from K nearest offset wells.
    # Leak-free: a well never sees its own data (self_wid is excluded).
    def __init__(self, k=12):
        self.k = k
        self.wells_xy = {}
        self.wells_forms = {}

    def fit(self, train_wids, data_dir):
        xs, ys, wids, forms = [], [], [], []
        for wid in train_wids:
            try:
                hw = pd.read_csv(data_dir / "train" / f"{wid}__horizontal_well.csv")
            except Exception:
                continue
            avail = [c for c in FORMATIONS if c in hw.columns]
            if not avail:
                continue
            med = hw[["X", "Y"] + avail].median(numeric_only=True)
            xs.append(med["X"]); ys.append(med["Y"]); wids.append(wid)
            forms.append([med.get(f, np.nan) for f in FORMATIONS])
        self.xy = np.array(list(zip(xs, ys)), dtype=float) if xs else np.zeros((0, 2))
        self.wids = np.array(wids)
        self.forms = np.array(forms, dtype=float) if forms else np.zeros((0, len(FORMATIONS)))
        self.tree = cKDTree(self.xy) if len(self.xy) else None
        return self

    def impute(self, xy_query, self_wid=None):
        n = len(xy_query)
        out = np.full((n, len(FORMATIONS)), np.nan, dtype=np.float32)
        dist_out = np.full(n, np.nan, dtype=np.float32)
        if self.tree is None or len(self.xy) < 3:
            return out, dist_out
        k = min(self.k + (5 if self_wid is not None else 0), len(self.xy))
        dists, idxs = self.tree.query(xy_query, k=k)
        if k == 1:
            dists = dists[:, None]; idxs = idxs[:, None]
        for i in range(n):
            ii = idxs[i]; dd = dists[i]
            if self_wid is not None:
                mask = self.wids[ii] != self_wid
                ii = ii[mask][: self.k]; dd = dd[mask][: self.k]
            else:
                ii = ii[: self.k]; dd = dd[: self.k]
            if len(ii) == 0:
                continue
            w = 1.0 / np.maximum(dd, 1e-3)
            w = w / w.sum()
            out[i] = (w[:, None] * self.forms[ii]).sum(axis=0)
            dist_out[i] = float(dd.min())
        return out, dist_out


class DenseANCCImputer:
    # Dense ANCC surface via inverse-distance weighted KNN over offset wells,
    # sampled at finer resolution along each well's trajectory.
    def __init__(self, k=15):
        self.k = k

    def fit(self, train_wids, data_dir):
        xs, ys, vs, wids = [], [], [], []
        for wid in train_wids:
            try:
                hw = pd.read_csv(data_dir / "train" / f"{wid}__horizontal_well.csv")
            except Exception:
                continue
            if "ANCC" not in hw.columns:
                continue
            sub = hw[["X", "Y", "ANCC"]].dropna()
            if len(sub) == 0:
                continue
            step = max(1, len(sub) // 40)
            sub = sub.iloc[::step]
            xs.extend(sub["X"].tolist()); ys.extend(sub["Y"].tolist()); vs.extend(sub["ANCC"].tolist())
            wids.extend([wid] * len(sub))
        self.xy = np.array(list(zip(xs, ys)), dtype=float) if xs else np.zeros((0, 2))
        self.v = np.array(vs, dtype=float)
        self.wids = np.array(wids)
        self.tree = cKDTree(self.xy) if len(self.xy) else None
        return self

    def impute(self, xy_query, self_wid=None):
        n = len(xy_query)
        out = np.full(n, np.nan, dtype=np.float32)
        std_out = np.full(n, np.nan, dtype=np.float32)
        dist_out = np.full(n, np.nan, dtype=np.float32)
        if self.tree is None or len(self.xy) < 3:
            return out, std_out, dist_out
        k = min(self.k + (10 if self_wid is not None else 0), len(self.xy))
        dists, idxs = self.tree.query(xy_query, k=k)
        if k == 1:
            dists = dists[:, None]; idxs = idxs[:, None]
        for i in range(n):
            ii = idxs[i]; dd = dists[i]
            if self_wid is not None:
                mask = self.wids[ii] != self_wid
                ii = ii[mask][: self.k]; dd = dd[mask][: self.k]
            else:
                ii = ii[: self.k]; dd = dd[: self.k]
            if len(ii) == 0:
                continue
            w = 1.0 / np.maximum(dd, 1e-3)
            w = w / w.sum()
            out[i] = float((w * self.v[ii]).sum())
            std_out[i] = float(self.v[ii].std())
            dist_out[i] = float(dd.min())
        return out, std_out, dist_out


def robust_slope(x, y, w=None):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    if len(x) < 3:
        return 0.0
    try:
        coef = np.polyfit(x - x[0], y, 1, w=w)
        return float(coef[0])
    except Exception:
        return 0.0


def affine_cal(kgr, tw_at_k, min_pts=20):
    if len(kgr) < min_pts:
        return 1.0, 0.0
    try:
        a, b = np.polyfit(tw_at_k, kgr, 1)
        return float(a), float(b)
    except Exception:
        return 1.0, 0.0


def seg_b_well(ktvt, kz, form_col):
    # Per-well offset (b) between known TVT+Z and a formation/surface
    # estimate, computed full / early / mid / late-segment and via weighted
    # least squares (down-weighting outliers).
    resid = ktvt + kz - form_col
    n = len(resid)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    b_full = float(np.nanmedian(resid))
    b_early = float(np.nanmedian(resid[: max(1, n // 3)]))
    b_mid = float(np.nanmedian(resid[n // 3: 2 * n // 3])) if n >= 3 else b_full
    b_late = float(np.nanmedian(resid[-max(1, n // 3):]))
    sc = 1.4826 * float(np.nanmedian(np.abs(resid - np.nanmedian(resid)))) + 1e-6
    w = 1.0 / (1.0 + ((resid - np.nanmedian(resid)) / (2.5 * sc)) ** 2)
    b_wls = float(np.nansum(w * resid) / max(np.nansum(w), 1e-9))
    return b_full, b_early, b_mid, b_late, b_wls


def multi_scale_ncc(kgr, ktvt, hgr, hws=(8, 15, 25), stride=3):
    # Normalised cross-correlation at multiple window scales between the
    # known-zone GR pattern and the evaluation-zone GR, producing a TVT path
    # candidate per scale plus a score-weighted ensemble.
    n_h = len(hgr)
    results = []
    for hw_size in hws:
        path = np.full(n_h, ktvt[-1] if len(ktvt) else 0.0, dtype=np.float32)
        scores = np.zeros(n_h, dtype=np.float32)
        if len(kgr) > hw_size and n_h > 0:
            template = kgr[-hw_size:]
            template_n = (template - template.mean()) / (template.std() + 1e-6)
            cur = ktvt[-1]
            step = max(1, (ktvt[-1] - ktvt[max(0, len(ktvt) - hw_size)]) / max(hw_size, 1)) if len(ktvt) > hw_size else 1.0
            for i in range(0, n_h, stride):
                lo, hi = i, min(i + hw_size, n_h)
                if hi - lo < hw_size // 2:
                    path[lo:hi] = cur
                    scores[lo:hi] = 0.0
                    continue
                window = hgr[lo:hi]
                window_n = (window - window.mean()) / (window.std() + 1e-6)
                L = min(len(window_n), len(template_n))
                ncc = float(np.dot(window_n[:L], template_n[:L]) / max(L, 1))
                drift = step * (i - 0) / max(hw_size, 1)
                cur_local = ktvt[-1] + drift
                path[lo:hi] = cur_local
                scores[lo:hi] = ncc
            cur = path[-1]
        results.append((path, scores))
    if results:
        all_paths = np.stack([r[0] for r in results], 0)
        all_scores = np.stack([np.clip(r[1], 0, None) for r in results], 0)
        wsum = all_scores.sum(0)
        wsum[wsum < 1e-6] = 1.0
        ens = (all_scores * all_paths).sum(0) / wsum
    else:
        ens = np.zeros(n_h, dtype=np.float32)
    return results, ens.astype(np.float32)



# %%
# ── per-well selector: which PF-scale / beam-hold variant to trust ────────
SELECTOR_N_EVAL_THRESHOLD = 4840.0
SELECTOR_Z_SPAN_THRESHOLDS = (136.73, 185.51)

SELECTOR_BIN_VARIANTS = {
    0: "pf_scale_5_hold_0.2", 1: "pf_scale_3_hold_0.15",
    2: "pf_scale_12_beam_0.2_hold_0.15", 3: "pf_scale_5_hold_0.15",
    4: "pf_scale_5_beam_0.05_hold_0.05", 5: "pf_scale_12_beam_0.2_hold_0.05",
}
SELECTOR_GLOBAL_VARIANT = "pf_scale_8_hold_0.2"


def selector_well_code(hw):
    ev = hw[hw["TVT_input"].isna()]
    z = hw["Z"]
    n_eval = len(ev)
    z_span = float(z.max() - z.min()) if len(z) else 0.0
    if n_eval < SELECTOR_N_EVAL_THRESHOLD:
        b = 0 if z_span < SELECTOR_Z_SPAN_THRESHOLDS[0] else 1
    else:
        b = 2 if z_span < SELECTOR_Z_SPAN_THRESHOLDS[1] else 3
    variant = SELECTOR_BIN_VARIANTS.get(b, SELECTOR_GLOBAL_VARIANT)
    return b, variant


def parse_selector_variant(name):
    # 'pf_scale_5_hold_0.2' or 'pf_scale_12_beam_0.2_hold_0.15' -> dict.
    parts = name.split("_")
    out = {"scale": 5.0, "beam": None, "hold": 0.0}
    try:
        si = parts.index("scale")
        out["scale"] = float(parts[si + 1])
    except (ValueError, IndexError):
        pass
    try:
        bi = parts.index("beam")
        out["beam"] = float(parts[bi + 1])
    except (ValueError, IndexError):
        pass
    try:
        hi = parts.index("hold")
        out["hold"] = float(parts[hi + 1])
    except (ValueError, IndexError):
        pass
    return out


def apply_selector_variant(name, pf_by_scale, tvt_beam, last_known_tvt):
    cfg = parse_selector_variant(name)
    key = f"scale_{cfg['scale']:g}"
    base = pf_by_scale.get(key, pf_by_scale.get("pf_mean"))
    if base is None:
        return None
    pred = base.copy()
    if cfg["beam"] is not None and tvt_beam is not None:
        pred = (1 - cfg["beam"]) * pred + cfg["beam"] * tvt_beam
    hold = cfg["hold"]
    if hold > 0:
        n_hold = int(hold * len(pred))
        if n_hold > 0:
            pred[:n_hold] = last_known_tvt
    return pred


print("Shared physics toolbox ready:",
      "PF + beam + spatial priors + selector + NCC + segment fits")


# %% [markdown]
# ---
# ### Projected-SP45 and shared feature surface
#
# Physics trackers, beam search, multi-scale NCC, and target-free spatial
# priors generate the visible source's projected-SP45 component and the
# feature surface consumed by the saved Pipeline-B boosters.
#

# %%
# ── numba-light single-particle-filter ANCC/Z trackers used inside the
#    feature builder (separate from the likelihood-ensemble PF above: these
#    are the cheaper single-pass trackers used as raw input signals) ───────
ANCC_N = 600
PF_N = 600
ANCH_OFFS = [-5, 0, 5]
BEAM_OFFS = [-5, 0, 5]
SC_OFFS = [-5, 0, 5]
PF_OFFS = [-5, 0, 5]
NCPU = CFG.n_jobs


def run_pf_ancc(hw, tw_tvt, tw_gr, well, split, N=ANCC_N):
    kn = hw[hw["TVT_input"].notna()]
    ev = hw[hw["TVT_input"].isna()]
    if len(ev) == 0 or len(kn) == 0:
        return np.array([]), np.array([])
    fake_tw = pd.DataFrame({"TVT": tw_tvt, "GR": tw_gr})
    pred, _ = run_particle_filter(
        hw,
        fake_tw,
        n_particles=N,
        seed=stable_seed(split, "feature_pf_ancc_primary", well, 0),
    )
    pred_ev = pred[list(ev.index)]
    # crude local std proxy from a second seed, for an uncertainty feature
    pred2, _ = run_particle_filter(
        hw,
        fake_tw,
        n_particles=max(100, N // 3),
        seed=stable_seed(split, "feature_pf_ancc_std", well, 0),
    )
    std_ev = np.abs(pred_ev - pred2[list(ev.index)]).astype(np.float32)
    return pred_ev.astype(np.float32), std_ev


def run_pf_z(hw, tw_tvt, tw_gr, well, split, N=PF_N):
    # Same tracker, alternate seed -> used as a decorrelated companion signal.
    kn = hw[hw["TVT_input"].notna()]
    ev = hw[hw["TVT_input"].isna()]
    if len(ev) == 0 or len(kn) == 0:
        return np.array([]), np.array([])
    fake_tw = pd.DataFrame({"TVT": tw_tvt, "GR": tw_gr})
    pred, _ = run_particle_filter(
        hw,
        fake_tw,
        n_particles=N,
        seed=stable_seed(split, "feature_pf_z", well, 0),
    )
    pred_ev = pred[list(ev.index)]
    return pred_ev.astype(np.float32), np.zeros_like(pred_ev, dtype=np.float32)


_FI = None  # FormationPlaneKNN, fit once on train wells before building features
_DI = None  # DenseANCCImputer


def init_imputers(train_wids, data_dir=None):
    global _FI, _DI
    data_dir = data_dir or CFG.DATA
    _FI = FormationPlaneKNN(k=12).fit(train_wids, data_dir)
    _DI = DenseANCCImputer(k=15).fit(train_wids, data_dir)


def build_well_A(hw_path, tw_path, is_train):
    # Pipeline-A per-well feature row builder. Produces one row per
    # evaluation-zone sample: tracker deltas, agreement/uncertainty, GR
    # statistics & residuals against the typewell at TVT offsets, geometry,
    # and spatial anchors.
    wid = Path(hw_path).stem.replace("__horizontal_well", "")
    try:
        hw = pd.read_csv(hw_path)
        tw = pd.read_csv(tw_path).sort_values("TVT")
    except Exception:
        return None
    if is_train and "TVT" not in hw.columns:
        return None
    kn = hw[hw["TVT_input"].notna()]
    ev = hw[hw["TVT_input"].isna()]
    if len(ev) == 0 or len(kn) < 10:
        return None
    if is_train and hw["TVT"].isna().all():
        return None
    tw_tvt = tw["TVT"].to_numpy(np.float32)
    tw_gr = tw["GR"].to_numpy(np.float32)
    if len(tw_tvt) < 3:
        return None

    split = "train" if is_train else "test"
    pf_a, std_a = run_pf_ancc(hw, tw_tvt, tw_gr, well=wid, split=split)
    if len(pf_a) == 0:
        return None
    pf_z, std_z = run_pf_z(hw, tw_tvt, tw_gr, well=wid, split=split)
    pf_use = pf_a.astype(np.float32)
    std_use = std_a.astype(np.float32)
    has_z = len(pf_z) == len(pf_a) and not np.any(np.isnan(pf_z))

    lk = kn.iloc[-1]
    last_tvt = float(lk["TVT_input"])
    gr_full = hw["GR"].astype(float).interpolate(limit_direction="both").fillna(float(np.nanmean(tw_gr)))
    hgr = gr_full.iloc[ev.index[0]:].to_numpy(np.float32)
    kgr = gr_full.iloc[: len(kn)].to_numpy(np.float32)

    bpaths = {}
    for (bs, mc, es, r, tag) in BEAMS:
        bpaths[tag] = beam_search(hgr, tw_tvt, tw_gr, last_tvt, bs, mc, es, r)
    beam_ref = (bpaths["cons"] + bpaths["sm5"]) / 2.0

    ktvt = kn["TVT_input"].to_numpy(np.float32)
    sc_res, sc_ens = multi_scale_ncc(kgr, ktvt, hgr, hws=(8, 15, 25), stride=3)
    sc8, sc8s = sc_res[0]; sc15, sc15s = sc_res[1]; sc25, sc25s = sc_res[2]
    sc_cons = (sc8 + sc15 + sc25) / 3.0
    sc_trust = float(np.clip(len(kn) / 200.0, 0.0, 0.6))
    hyb_ref = (1 - sc_trust) * beam_ref + sc_trust * sc_ens

    tw_at_k = np.interp(ktvt, tw_tvt, tw_gr).astype(np.float32)
    a_cal, b_cal = affine_cal(kgr, tw_at_k)
    kmd = kn["MD"].to_numpy(np.float32); kz = kn["Z"].to_numpy(np.float32)
    pfx_rmse = float(np.sqrt(np.mean((kgr - tw_at_k) ** 2)))
    slp_all = robust_slope(kmd, ktvt); slp_50 = robust_slope(kmd[-50:], ktvt[-50:])
    slp_z = robust_slope(kz, ktvt)

    # Exclude a matching train well for both train and test construction.
    # This blocks any same-ID spatial lookup while retaining target-free KNN.
    swid = wid
    xy_ev = ev[["X", "Y"]].to_numpy(np.float64)
    xy_kn = kn[["X", "Y"]].to_numpy(np.float64)
    form_ev, knn_d = _FI.impute(xy_ev, self_wid=swid)
    form_kn, _ = _FI.impute(xy_kn, self_wid=swid)
    z_kn = kn["Z"].to_numpy(np.float32); z_ev = ev["Z"].to_numpy(np.float32)

    tvt_fs = {}; form_rmse = {}; form_list = []
    for fi2, fn in enumerate(FORMATIONS):
        b_full, b_early, b_mid, b_late, b_wls = seg_b_well(ktvt, z_kn, form_kn[:, fi2])
        tvt_f = (-z_ev + form_ev[:, fi2] + b_full).astype(np.float32)
        tvt_fw = (-z_ev + form_ev[:, fi2] + b_wls).astype(np.float32)
        tvt_f50 = (-z_ev + form_ev[:, fi2] + b_late).astype(np.float32)
        tvt_fs[f"tvtF_{fn}"] = tvt_f; tvt_fs[f"tvtFw_{fn}"] = tvt_fw; tvt_fs[f"tvtF50_{fn}"] = tvt_f50
        tvt_fs[f"bw_{fn}"] = np.float32(b_full); tvt_fs[f"bww_{fn}"] = np.float32(b_wls)
        tvt_fs[f"bw50_{fn}"] = np.float32(b_late)
        tvt_fs[f"bw_early_{fn}"] = np.float32(b_early); tvt_fs[f"bw_mid_{fn}"] = np.float32(b_mid)
        form_rmse[fn] = float(np.sqrt(np.mean((ktvt - (-z_kn + form_kn[:, fi2] + b_full)) ** 2)))
        form_list.append(tvt_f)

    fs = np.stack(form_list, 1)
    form_mean_d = (fs.mean(1) - last_tvt).astype(np.float32)
    form_std_d = fs.std(1).astype(np.float32)
    form_rng_d = (fs.max(1) - fs.min(1)).astype(np.float32)

    d_ancc, d_std, d_dist = _DI.impute(xy_ev, self_wid=swid)
    d_kn, d_std_kn, _ = _DI.impute(xy_kn, self_wid=swid)
    b_vd = ktvt + z_kn - d_kn
    _, b_de, b_dm, b_dl, b_dw = seg_b_well(ktvt, z_kn, d_kn)
    b_d = float(np.nanmedian(b_vd)) if np.isfinite(b_vd).any() else 0.0
    tvt_dense = (-z_ev + d_ancc + b_d).astype(np.float32)
    tvt_densew = (-z_ev + d_ancc + b_dw).astype(np.float32)
    tvt_dense50 = (-z_ev + d_ancc + b_dl).astype(np.float32)
    res_kn = ktvt + z_kn - d_kn
    d_rmse = float(np.sqrt(np.nanmean(res_kn ** 2))) if np.isfinite(res_kn).any() else 0.0
    d_bias = float(np.nanmean(res_kn)) if np.isfinite(res_kn).any() else 0.0
    d_nb_std = float(np.nanmean(d_std_kn)) if np.isfinite(d_std_kn).any() else 0.0

    all_sigs = [pf_use] + [p for p in bpaths.values()] + [sc8, sc15, sc25, sc_ens, tvt_fs["tvtF_ANCC"], tvt_dense]
    sig_mat = np.stack(all_sigs, 1)
    sig_std = sig_mat.std(1).astype(np.float32)
    sig_mean = (sig_mat.mean(1) - last_tvt).astype(np.float32)

    gr_s = pd.Series(gr_full.values); rolls = {}
    for w in [5, 21, 51, 101]:
        r = gr_s.rolling(w, center=True, min_periods=1)
        rolls[f"grm{w}"] = r.mean().iloc[ev.index].values.astype(np.float32)
        rolls[f"grs{w}"] = r.std().fillna(0).iloc[ev.index].values.astype(np.float32)
    for lag in [1, 5, 15, 30]:
        rolls[f"glag{lag}"] = gr_s.shift(lag).bfill().iloc[ev.index].values.astype(np.float32)
        rolls[f"glead{lag}"] = gr_s.shift(-lag).ffill().iloc[ev.index].values.astype(np.float32)
    gr_d1 = gr_s.diff().fillna(0.0).iloc[ev.index].values.astype(np.float32)
    gr_d2 = gr_s.diff().diff().fillna(0.0).iloc[ev.index].values.astype(np.float32)
    gr_env = gr_s.rolling(21, center=True, min_periods=1).max().iloc[ev.index].values.astype(np.float32)
    gr_nrg = np.sqrt(np.maximum((gr_s ** 2).rolling(21, center=True, min_periods=1).mean(), 0.0)).iloc[ev.index].values.astype(np.float32)

    hmd = ev["MD"].to_numpy(np.float32); md_since = hmd - float(lk["MD"])
    slp_b_all = (last_tvt + slp_all * md_since).astype(np.float32)
    slp_b_50 = (last_tvt + slp_50 * md_since).astype(np.float32)

    mdd = hw["MD"].diff().replace(0, np.nan)
    dzdmd = (hw["Z"].diff() / mdd).iloc[ev.index].values.astype(np.float32)
    dxdmd = (hw["X"].diff() / mdd).iloc[ev.index].values.astype(np.float32)
    dydmd = (hw["Y"].diff() / mdd).iloc[ev.index].values.astype(np.float32)

    nh = len(ev); frac = (np.arange(nh) / max(nh - 1, 1)).astype(np.float32)
    def sc(v): return np.full(nh, np.float32(v), np.float32)

    feats = {
        "well": wid, "id": [f"{wid}_{i}" for i in ev.index],
        "last_known_tvt": sc(last_tvt),
        "pf_ancc": pf_use, "pf_ancc_std": std_use,
        "pf_ancc_delta": (pf_use - last_tvt).astype(np.float32),
        "pf_z": (pf_z.astype(np.float32) if has_z else sc(last_tvt)),
        "pf_z_delta": ((pf_z - last_tvt).astype(np.float32) if has_z else sc(0.0)),
        "pf_vs_z": ((pf_use - pf_z.astype(np.float32)) if has_z else sc(0.0)),
        **{f"beam_{t}_d": (p - np.float32(last_tvt)).astype(np.float32) for t, p in bpaths.items()},
        "beam_mean_d": np.stack([(p - last_tvt) for p in bpaths.values()], 1).mean(1).astype(np.float32),
        "beam_std_d": np.stack([(p - last_tvt) for p in bpaths.values()], 1).std(1).astype(np.float32),
        "beam_med_d": np.median(np.stack([(p - last_tvt) for p in bpaths.values()], 1), 1).astype(np.float32),
        "sc8_d": (sc8 - np.float32(last_tvt)).astype(np.float32), "sc8_sc": sc8s,
        "sc15_d": (sc15 - np.float32(last_tvt)).astype(np.float32), "sc15_sc": sc15s,
        "sc25_d": (sc25 - np.float32(last_tvt)).astype(np.float32), "sc25_sc": sc25s,
        "sc_cons_d": (sc_cons - np.float32(last_tvt)).astype(np.float32),
        "sc_ens_d": (sc_ens - np.float32(last_tvt)).astype(np.float32),
        "sc_trust": sc(sc_trust), "hyb_d": (hyb_ref - np.float32(last_tvt)).astype(np.float32),
        "sig_std": sig_std, "sig_mean_d": sig_mean,
        **tvt_fs,
        **{f"frm_rmse_{fn}": sc(form_rmse[fn]) for fn in FORMATIONS},
        "form_mean_d": form_mean_d, "form_std_d": form_std_d, "form_rng_d": form_rng_d,
        "spatial_ancc_d": (form_ev[:, 0] - np.float32(np.interp(last_tvt, tw_tvt, tw_gr))),
        "spatial_knn_dist": knn_d,
        "dense_ancc": d_ancc, "dense_std": d_std, "dense_dist": d_dist,
        "tvt_dense_d": (tvt_dense - last_tvt).astype(np.float32),
        "tvt_densew_d": (tvt_densew - last_tvt).astype(np.float32),
        "tvt_dense50_d": (tvt_dense50 - last_tvt).astype(np.float32),
        "dense_rmse": sc(d_rmse), "dense_bias": sc(d_bias), "dense_nb_std": sc(d_nb_std),
        "pf_vs_spatial": (pf_use - tvt_fs["tvtF_ANCC"]).astype(np.float32),
        "pf_vs_dense": (pf_use - tvt_dense).astype(np.float32),
        "spatial_vs_dense": (tvt_fs["tvtF_ANCC"] - tvt_dense).astype(np.float32),
        "beam_vs_spatial": (bpaths["cons"] - tvt_fs["tvtF_ANCC"]).astype(np.float32),
        "sc_vs_beam": (sc_ens - bpaths["cons"]).astype(np.float32),
        "cal_a": sc(a_cal), "cal_b": sc(b_cal),
        "pfx_rmse": sc(pfx_rmse), "known_len": sc(len(kn)), "eval_len": sc(nh),
        "slp_all": sc(slp_all), "slp_50": sc(slp_50), "slp_z": sc(slp_z),
        "slp_b_d_all": (slp_b_all - last_tvt).astype(np.float32),
        "slp_b_d_50": (slp_b_50 - last_tvt).astype(np.float32),
        "ktvt_range": sc(float(np.ptp(ktvt))), "ktvt_std": sc(float(ktvt.std())),
        "md_since": md_since, "frac": frac, "frac2": frac ** 2, "sqrt_frac": np.sqrt(frac),
        "z": z_ev,
        "dx": (ev["X"] - float(lk["X"])).to_numpy(np.float32),
        "dy": (ev["Y"] - float(lk["Y"])).to_numpy(np.float32),
        "dz": (z_ev - float(lk["Z"])).astype(np.float32),
        "dxy": np.sqrt((ev["X"] - float(lk["X"])) ** 2 + (ev["Y"] - float(lk["Y"])) ** 2).to_numpy(np.float32),
        "dzdmd": dzdmd, "dxdmd": dxdmd, "dydmd": dydmd,
        "gr": hgr, "gr_d1": gr_d1, "gr_d2": gr_d2, "gr_env": gr_env, "gr_nrg": gr_nrg,
        "gr_vs_tw_anc": hgr - np.float32(np.interp(last_tvt, tw_tvt, tw_gr)),
        "gr_vs_slp_all": hgr - np.interp(slp_b_all, tw_tvt, tw_gr).astype(np.float32),
        **{f"tda{int(o)}": hgr - np.float32(np.interp(last_tvt + o, tw_tvt, tw_gr)) for o in ANCH_OFFS},
        **{f"tdbc{int(o)}": hgr - np.interp(beam_ref + o, tw_tvt, tw_gr).astype(np.float32) for o in BEAM_OFFS},
        **{f"tdsc{int(o)}": hgr - np.interp(sc_ens + o, tw_tvt, tw_gr).astype(np.float32) for o in SC_OFFS},
        **{f"tdpf{int(o)}": hgr - np.interp(pf_use + o, tw_tvt, tw_gr).astype(np.float32) for o in PF_OFFS},
        "tw_range": sc(float(np.ptp(tw_tvt))), "tw_gr_mean": sc(float(tw_gr.mean())),
    }
    for k, v in rolls.items():
        feats[k] = v
    result = pd.DataFrame(feats)
    if is_train:
        if "TVT" not in ev.columns or ev["TVT"].isna().all():
            return None
        result["target"] = (ev["TVT"].to_numpy(np.float32) - np.float32(last_tvt))
    return result


def build_dataset_A(paths, is_train):
    args = [(str(p), str(p.parent / f'{p.stem.replace("__horizontal_well", "")}__typewell.csv'), is_train)
            for p in paths if (p.parent / f'{p.stem.replace("__horizontal_well", "")}__typewell.csv').exists()]
    res = Parallel(n_jobs=NCPU, prefer="threads", verbose=3)(
        delayed(build_well_A)(hp, tp, it) for hp, tp, it in args)
    parts = [r for r in res if r is not None]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()



# %% [markdown]
# ---
# ### Pipeline-B likelihood features and fixed post-process
#
# An **independent implementation path** of the same physics: this engine
# adds **likelihood-weighted multi-scale particle filter features**
# (`likpf_scale_3/5/8/12`, computed once per well and merged onto the same
# feature table used by projected SP45) on top of the shared toolbox. exp510
# only applies the SHA-pinned saved boosters and preserves the visible source's
# fixed drift-aware post-process.
#

# %%
# ── lik-PF features (the addition that distinguishes Pipeline B) ──────────
def likpf_rows_for_well(wid, split):
    hw, tw = load_well(wid, split=split)
    ev_idx = hw.index[hw["TVT_input"].isna()]
    if len(ev_idx) == 0:
        return None
    pf_by_scale = get_likpf_bank(wid, split, hw, tw)
    d = {"id": [f"{wid}_{i}" for i in ev_idx]}
    for key, arr in pf_by_scale.items():
        col = "likpf_" + key.replace("pf_scale_", "scale_").replace("pf_mean", "mean")
        d[col] = arr[ev_idx].astype(np.float32)
    return pd.DataFrame(d)


def build_likpf(wids, split):
    res = Parallel(n_jobs=CFG.n_jobs, prefer="threads")(
        delayed(likpf_rows_for_well)(w, split) for w in wids)
    parts = [r for r in res if r is not None]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["id"])


def add_likpf_features(df, likpf):
    df = df.merge(likpf, on="id", how="left")
    for c in [c for c in likpf.columns if c != "id"]:
        df[c] = df[c].fillna(df["last_known_tvt"])
        df[c + "_d"] = (df[c] - df["last_known_tvt"]).astype(np.float32)
    return df



# %%
# ── drift-aware post-process: tuned recipe ─────────────────────────────────
class PP_B:
    alpha = 1.0
    tau = 85.0
    w_pf = 0.0
    w_sub1 = 0.60   # weight on the learned model; lik-PF gets 1-w_sub1 (flat CV optimum 0.55-0.68)
    sub2_scale = "scale_5"
    sg_win = 61
    sg_poly = 3


def warmup_B(md_since, tau):
    return 1.0 - np.exp(-np.maximum(md_since, 0.0) / tau) if tau > 1e-6 else 1.0


def make_prediction_B(df, model_delta):
    last = df["last_known_tvt"].values.astype(float)
    pf_delta = df["pf_ancc"].values.astype(float) - last
    lp_col = f"likpf_{PP_B.sub2_scale}"
    lp = df[lp_col].values.astype(float) - last if lp_col in df.columns else pf_delta
    sub1 = PP_B.alpha * warmup_B(df["md_since"].values.astype(float), PP_B.tau) * (model_delta * (1 - PP_B.w_pf) + pf_delta * PP_B.w_pf)
    delta = PP_B.w_sub1 * sub1 + (1 - PP_B.w_sub1) * lp
    pred = last + delta
    out = pred.copy()
    dfx = df.reset_index(drop=True)
    for _, idx in dfx.groupby("well", sort=False).groups.items():
        pos = dfx.index.get_indexer(idx)
        v = pred[pos]; n = len(v); wl = min(PP_B.sg_win, n)
        if wl % 2 == 0:
            wl -= 1
        if wl >= PP_B.sg_poly + 2:
            out[pos] = savgol_filter(v, wl, PP_B.sg_poly)
    return out


# %% [markdown]
# ## 4. Setup and input contract

# %%
def validate_sample(sample):
    required = {"id", "tvt"}
    if not required.issubset(sample.columns):
        raise RuntimeError(f"sample submission is missing columns: {sorted(required - set(sample.columns))}")
    result = sample[["id", "tvt"]].copy()
    result["id"] = result["id"].astype(str)
    if result.empty:
        raise RuntimeError("sample submission is empty")
    if result["id"].duplicated().any():
        raise RuntimeError("sample submission contains duplicate IDs")
    wells = result["id"].str.rsplit("_", n=1).str[0]
    if wells.eq("").any() or wells.nunique() < 1:
        raise RuntimeError("sample-derived well IDs are empty")
    return result


def validate_component(frame, sample, label, value_column="tvt"):
    required = {"id", value_column}
    if not required.issubset(frame.columns):
        raise RuntimeError(f"{label} is missing columns: {sorted(required - set(frame.columns))}")
    component = frame[["id", value_column]].copy()
    component["id"] = component["id"].astype(str)
    component[value_column] = component[value_column].astype(np.float64)
    if component["id"].duplicated().any():
        raise RuntimeError(f"{label} contains duplicate IDs")
    if set(component["id"]) != set(sample["id"]):
        missing = sorted(set(sample["id"]) - set(component["id"]))[:5]
        extra = sorted(set(component["id"]) - set(sample["id"]))[:5]
        raise RuntimeError(f"{label} ID mismatch: missing={missing}, extra={extra}")
    ordered = sample[["id"]].merge(component, on="id", how="left", validate="one_to_one")
    values = ordered[value_column].to_numpy(np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError(f"{label} contains non-finite predictions")
    return ordered.rename(columns={value_column: "tvt"})


def exact_public_preoverride(sp45, pipeline_b):
    if not np.array_equal(sp45["id"].to_numpy(), pipeline_b["id"].to_numpy()):
        raise RuntimeError("public component inputs are not in identical ID order")
    result = sp45[["id"]].copy()
    result["tvt"] = (
        SP45_WEIGHT * sp45["tvt"].to_numpy(np.float64)
        + PIPELINE_B_WEIGHT * pipeline_b["tvt"].to_numpy(np.float64)
    )
    return result


def exact_final_hedge(exp413, public_component):
    if not np.array_equal(exp413["id"].to_numpy(), public_component["id"].to_numpy()):
        raise RuntimeError("final component inputs are not in identical ID order")
    result = exp413[["id"]].copy()
    result["tvt"] = (
        EXP413_WEIGHT * exp413["tvt"].to_numpy(np.float64)
        + PUBLIC_WEIGHT * public_component["tvt"].to_numpy(np.float64)
    )
    return result


_LIKPF_BANK = {}


def get_likpf_bank(wid, split, hw, tw):
    key = (str(split), str(wid))
    if key not in _LIKPF_BANK:
        _LIKPF_BANK[key] = run_pf_lik_ensemble_scales(
            hw,
            tw,
            well=wid,
            split=split,
            family="public_likpf_bank",
            scales=SELECTOR_SCALES,
            n_particles=350,
            n_seeds=48,
        )
    return _LIKPF_BANK[key]


# %% [markdown]
# ## 5. Exact projected-SP45 component generation

# %%
def selector_predict_for_well(wid, hw, tw, split="test"):
    pf_by_scale = get_likpf_bank(wid, split, hw, tw)
    try:
        tvt_beam = run_beam_ensemble(hw, tw)
    except Exception:
        tvt_beam = pf_by_scale.get("pf_mean")
    _, variant = selector_well_code(hw)
    kn = hw[hw["TVT_input"].notna()]
    last_known_tvt = float(kn["TVT_input"].iloc[-1]) if len(kn) else 0.0
    prediction = apply_selector_variant(variant, pf_by_scale, tvt_beam, last_known_tvt)
    if prediction is None:
        prediction = pf_by_scale["pf_mean"]
    output = hw["TVT_input"].to_numpy(np.float64).copy()
    eval_index = hw.index[hw["TVT_input"].isna()]
    output[eval_index] = prediction[eval_index] if len(prediction) == len(hw) else prediction
    return output


def robust_projection_fit(md, values, degree=4):
    md = np.asarray(md, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    mask = np.isfinite(md) & np.isfinite(values)
    md = md[mask]
    values = values[mask]
    if len(md) < degree + 2:
        return None
    origin = md[0]
    scale = md.max() - md.min() if md.max() > md.min() else 1.0
    x_norm = (md - origin) / scale
    coefficients = np.polyfit(x_norm, values, degree)
    for _ in range(6):
        fitted = np.polyval(coefficients, x_norm)
        residual = values - fitted
        robust_scale = 1.4826 * np.median(np.abs(residual - np.median(residual))) + 1e-6
        weights = 1.0 / (1.0 + (residual / (4.685 * robust_scale)) ** 2)
        coefficients = np.polyfit(x_norm, values, degree, w=weights)
    return coefficients, origin, scale


def project_sp45_well(hw, selector_eval_prediction):
    known = hw[hw["TVT_input"].notna()]
    evaluation = hw[hw["TVT_input"].isna()]
    anchor = float(known["TVT_input"].iloc[-1])
    known_u = known["TVT_input"].to_numpy(np.float64) + known["Z"].to_numpy(np.float64) - anchor
    fitted = robust_projection_fit(known["MD"].to_numpy(np.float64), known_u, degree=4)
    if fitted is None:
        return np.asarray(selector_eval_prediction, dtype=np.float64)
    coefficients, origin, scale = fitted
    eval_u = np.polyval(coefficients, (evaluation["MD"].to_numpy(np.float64) - origin) / scale)
    projected = eval_u + anchor - evaluation["Z"].to_numpy(np.float64)
    return 0.75 * projected + 0.25 * np.asarray(selector_eval_prediction, dtype=np.float64)


def build_sp45_component(test_paths, sample):
    rows = []
    for horizontal_path in test_paths:
        wid = horizontal_path.stem.replace("__horizontal_well", "")
        hw, tw = load_well(wid, split="test")
        known_count = int(hw["TVT_input"].notna().sum())
        if known_count < 5:
            raise RuntimeError(f"SP45 cannot evaluate {wid}: known prefix rows={known_count}")
        selector_prediction = selector_predict_for_well(wid, hw, tw, split="test")
        eval_index = hw.index[hw["TVT_input"].isna()]
        projected = project_sp45_well(hw, selector_prediction[eval_index])
        rows.append(
            pd.DataFrame(
                {
                    "id": [f"{wid}_{row_index}" for row_index in eval_index],
                    "tvt": projected.astype(np.float64),
                }
            )
        )
    if not rows:
        raise RuntimeError("SP45 generated no rows")
    raw = pd.concat(rows, ignore_index=True)
    return validate_component(raw, sample, "projected SP45")


# %% [markdown]
# ## 6. Saved Pipeline-B inference

# %%
def build_pipeline_b_component(test_features, test_wids, sample, model_dir):
    likelihood_features = build_likpf(test_wids, "test")
    enriched = add_likpf_features(test_features.copy(), likelihood_features).reset_index(drop=True)
    feature_names = json.loads((model_dir / "features.json").read_text())
    if len(feature_names) != len(set(feature_names)):
        raise RuntimeError("Pipeline-B feature schema contains duplicates")
    missing = set(feature_names) - set(enriched.columns)
    unexpected_missing = missing - EXPECTED_ZERO_FILLED_FEATURES
    if unexpected_missing:
        raise RuntimeError(f"unexpected Pipeline-B feature gaps: {sorted(unexpected_missing)}")
    for name in sorted(missing):
        enriched[name] = np.float32(0.0)
    matrix = enriched[feature_names].to_numpy(np.float32)
    if not np.isfinite(matrix).all():
        raise RuntimeError("Pipeline-B feature matrix contains non-finite values")
    models = [joblib.load(model_dir / f"lgb{index}.pkl") for index in range(3)]
    model_delta = np.mean([model.predict(matrix) for model in models], axis=0)
    prediction = make_prediction_B(enriched, model_delta)
    raw = enriched[["id"]].copy()
    raw["tvt"] = prediction.astype(np.float64)
    component = validate_component(raw, sample, "Pipeline B")
    feature_audit = {
        "feature_count": len(feature_names),
        "rows": len(enriched),
        "zero_filled_features": sorted(missing),
        "feature_schema_sha256": hashlib.sha256(
            json.dumps(feature_names, separators=(",", ":")).encode()
        ).hexdigest(),
        "feature_content_sha256": dataframe_content_sha(enriched, ["id", *feature_names]),
        "likpf_schema_sha256": schema_sha(likelihood_features),
        "likpf_content_sha256": dataframe_content_sha(
            likelihood_features,
            list(likelihood_features.columns),
        ),
    }
    return component, enriched, feature_audit


# %% [markdown]
# ## 7. Dynamic hidden-safe exp413 generation and fixed hedge

# %%
def reload_dynamic_exp413_artifact(path, in_memory_frame):
    """Reload the generated CSV boundary used by the original exp510 formula.

    exp413 stores ``pred_tvt`` as float32.  Blending that in-memory array directly
    is not numerically identical to consuming the generated CSV, which is the
    component boundary used by exp510 before the hidden-safe repair.
    """
    path = Path(path)
    if not path.is_file():
        raise RuntimeError(f"dynamic exp413 prediction artifact is missing: {path}")
    required = {"id", "well", "last_known_tvt", "pred_tvt"}
    serialized = pd.read_csv(path)
    for label, frame in (("in-memory", in_memory_frame), ("serialized", serialized)):
        missing = required - set(frame.columns)
        if missing:
            raise RuntimeError(f"{label} exp413 prediction is missing columns: {sorted(missing)}")
        if frame["id"].astype(str).duplicated().any():
            raise RuntimeError(f"{label} exp413 prediction contains duplicate IDs")

    memory_values = in_memory_frame[["id", "pred_tvt"]].copy()
    memory_values["id"] = memory_values["id"].astype(str)
    serialized["id"] = serialized["id"].astype(str)
    comparison = serialized[["id", "pred_tvt"]].merge(
        memory_values,
        on="id",
        how="outer",
        suffixes=("_serialized", "_memory"),
        indicator=True,
        validate="one_to_one",
    )
    if not comparison["_merge"].eq("both").all():
        raise RuntimeError("serialized exp413 prediction IDs differ from the in-memory result")
    serialized_values = comparison["pred_tvt_serialized"].to_numpy(np.float64)
    memory_array = comparison["pred_tvt_memory"].to_numpy(np.float64)
    if not np.isfinite(serialized_values).all() or not np.isfinite(memory_array).all():
        raise RuntimeError("exp413 serialization boundary contains non-finite predictions")
    max_abs = float(np.max(np.abs(serialized_values - memory_array)))
    if max_abs > 1e-3:
        raise RuntimeError(f"exp413 serialization roundtrip drift is too large: {max_abs}")
    return serialized, max_abs


def load_exp413_component(frame, sample):
    if "well" not in frame.columns or "pred_tvt" not in frame.columns:
        raise RuntimeError("exp413 prediction is missing well/pred_tvt")
    component = validate_component(frame, sample, "exp413", value_column="pred_tvt")
    context = frame[["id", "well", "last_known_tvt"]].copy()
    context["id"] = context["id"].astype(str)
    context = sample[["id"]].merge(context, on="id", how="left", validate="one_to_one")
    return component, context


def difference_metrics(left, right):
    delta = left.to_numpy(np.float64) - right.to_numpy(np.float64)
    absolute = np.abs(delta)
    return {
        "rmse": float(np.sqrt(np.mean(delta**2))),
        "mae": float(np.mean(absolute)),
        "mean_signed": float(np.mean(delta)),
        "p95_abs": float(np.quantile(absolute, 0.95)),
        "max_abs": float(np.max(absolute)),
    }


def grouped_difference_readout(frame, group_column):
    rows = []
    comparisons = {
        "sp45_minus_pipeline_b": ("sp45", "pipeline_b"),
        "public_minus_exp413": ("public_preoverride", "exp413"),
        "final_minus_exp413": ("final", "exp413"),
    }
    for group_value, group in frame.groupby(group_column, sort=True, observed=True):
        row = {group_column: str(group_value), "rows": int(len(group))}
        for label, (left, right) in comparisons.items():
            metrics = difference_metrics(group[left], group[right])
            row.update({f"{label}_{name}": value for name, value in metrics.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def build_readouts(
    test_features,
    exp413_context,
    sp45,
    pipeline_b,
    public_component,
    exp413,
    final,
):
    context = test_features[["id", "well", "md_since"]].copy()
    context["id"] = context["id"].astype(str)
    context = context.merge(exp413_context, on=["id", "well"], how="left", validate="one_to_one")
    table = context.merge(sp45.rename(columns={"tvt": "sp45"}), on="id", validate="one_to_one")
    table = table.merge(
        pipeline_b.rename(columns={"tvt": "pipeline_b"}),
        on="id",
        validate="one_to_one",
    )
    table = table.merge(
        public_component.rename(columns={"tvt": "public_preoverride"}),
        on="id",
        validate="one_to_one",
    )
    table = table.merge(exp413.rename(columns={"tvt": "exp413"}), on="id", validate="one_to_one")
    table = table.merge(final.rename(columns={"tvt": "final"}), on="id", validate="one_to_one")
    table["horizon_bucket"] = pd.cut(
        table["md_since"].astype(float),
        bins=[-np.inf, 250.0, 500.0, 1000.0, 2000.0, np.inf],
        labels=["0000_0250", "0250_0500", "0500_1000", "1000_2000", "2000_plus"],
        right=False,
    )
    overall = {
        "sp45_vs_pipeline_b": difference_metrics(table["sp45"], table["pipeline_b"]),
        "public_vs_exp413": difference_metrics(table["public_preoverride"], table["exp413"]),
        "final_vs_exp413": difference_metrics(table["final"], table["exp413"]),
    }
    first_rows = table.sort_values(["well", "md_since"]).groupby("well", sort=False).head(1)
    start_delta = first_rows["final"].to_numpy(np.float64) - first_rows["last_known_tvt"].to_numpy(np.float64)
    overall["start_continuity"] = {
        "wells": int(len(first_rows)),
        "mean_abs_ft": float(np.mean(np.abs(start_delta))),
        "p95_abs_ft": float(np.quantile(np.abs(start_delta), 0.95)),
        "max_abs_ft": float(np.max(np.abs(start_delta))),
    }
    return (
        table,
        grouped_difference_readout(table, "well"),
        grouped_difference_readout(table, "horizon_bucket"),
        overall,
    )


# %% [markdown]
# ## 8. Metrics, diagnostics, and generated artifacts

# %%
def run_inference():
    started = time.time()
    CFG.DATA = find_data_dir()
    CFG.OUT.mkdir(parents=True, exist_ok=True)
    sample = validate_sample(pd.read_csv(CFG.DATA / "sample_submission.csv"))
    model_dir = find_exact_model_dir()
    exp413_frame_memory, exp413_runtime_metrics, exp413_path = (
        generate_dynamic_exp413_prediction()
    )
    if int(exp413_runtime_metrics.get("booster_training_count", -1)) != 0:
        raise RuntimeError("dynamic exp413 runtime trained an unexpected booster")
    if bool(exp413_runtime_metrics.get("external_submission_performed", True)):
        raise RuntimeError("dynamic exp413 runtime performed an external submission")
    if int(exp413_runtime_metrics.get("rows", -1)) != len(sample):
        raise RuntimeError("dynamic exp413 runtime row count differs from sample")
    exp413_frame, exp413_serialization_roundtrip_max_abs = (
        reload_dynamic_exp413_artifact(exp413_path, exp413_frame_memory)
    )

    train_wids = sorted(
        path.stem.replace("__horizontal_well", "")
        for path in (CFG.DATA / "train").glob("*__horizontal_well.csv")
    )
    test_paths = sorted((CFG.DATA / "test").glob("*__horizontal_well.csv"))
    test_wids = [path.stem.replace("__horizontal_well", "") for path in test_paths]
    if not train_wids or not test_wids:
        raise RuntimeError("competition train/test wells are empty")
    print(
        json.dumps(
            {
                "route": "ensemble",
                "source": SOURCE_KERNEL_ID,
                "source_sha256": SOURCE_SHA256,
                "rows": len(sample),
                "test_wells": len(test_wids),
                "model_dataset": MODEL_DATASET,
                "model_dataset_version": MODEL_DATASET_VERSION,
                "new_models": 0,
                "boosters_loaded": 78,
                "exp413_saved_boosters": 75,
                "pipeline_b_saved_boosters": 3,
            },
            indent=2,
        ),
        flush=True,
    )

    init_imputers(train_wids, CFG.DATA)
    test_features = build_dataset_A(test_paths, is_train=False).reset_index(drop=True)
    if test_features.empty or test_features["id"].duplicated().any():
        raise RuntimeError("test feature generation returned empty or duplicate IDs")
    if set(test_features["id"].astype(str)) != set(sample["id"]):
        raise RuntimeError("test feature IDs do not cover the dynamic sample exactly")

    sp45 = build_sp45_component(test_paths, sample)
    pipeline_b, pipeline_b_features, feature_audit = build_pipeline_b_component(
        test_features,
        test_wids,
        sample,
        model_dir,
    )
    public_component = exact_public_preoverride(sp45, pipeline_b)
    exp413, exp413_context = load_exp413_component(exp413_frame, sample)
    final = exact_final_hedge(exp413, public_component)

    public_parity = np.max(
        np.abs(
            public_component["tvt"].to_numpy(np.float64)
            - (SP45_WEIGHT * sp45["tvt"] + PIPELINE_B_WEIGHT * pipeline_b["tvt"]).to_numpy(np.float64)
        )
    )
    final_parity = np.max(
        np.abs(
            final["tvt"].to_numpy(np.float64)
            - (EXP413_WEIGHT * exp413["tvt"] + PUBLIC_WEIGHT * public_component["tvt"]).to_numpy(np.float64)
        )
    )
    if public_parity > 1e-12 or final_parity > 1e-12:
        raise RuntimeError(f"blend formula parity failed: public={public_parity}, final={final_parity}")

    sp45_path = CFG.OUT / "sp45_projection_submission.csv"
    pipeline_b_path = CFG.OUT / "submission_B.csv"
    public_path = CFG.OUT / "public_preoverride_submission.csv"
    submission_path = CFG.OUT / "submission.csv"
    sp45.to_csv(sp45_path, index=False)
    pipeline_b.to_csv(pipeline_b_path, index=False)
    public_component.to_csv(public_path, index=False)
    final.to_csv(submission_path, index=False)

    component_table, by_well, by_horizon, readout = build_readouts(
        test_features,
        exp413_context,
        sp45,
        pipeline_b,
        public_component,
        exp413,
        final,
    )
    by_well.to_csv(CFG.OUT / "exp510_by_well_readout.csv", index=False)
    by_horizon.to_csv(CFG.OUT / "exp510_horizon_readout.csv", index=False)
    component_table[[
        "id",
        "well",
        "md_since",
        "sp45",
        "pipeline_b",
        "public_preoverride",
        "exp413",
        "final",
    ]].to_csv(CFG.OUT / "exp510_component_readout.csv.gz", index=False, compression="gzip")

    model_sha = {name: sha256_file(model_dir / name) for name in MODEL_SHA256}
    manifest = {
        "status": "technical_gate_pass",
        "source_kernel_id": SOURCE_KERNEL_ID,
        "source_sha256": SOURCE_SHA256,
        "source_boundary": "visible_final_blend_output_before_excluded_stages",
        "model_dataset": MODEL_DATASET,
        "model_dataset_version": MODEL_DATASET_VERSION,
        "model_sha256": model_sha,
        "exp413_prediction": {
            "generation": "dynamic_hidden_safe_exp413_v4_runtime_serialized_artifact_boundary",
            "parent_source_sha256": EXP413_PARENT_SOURCE_SHA256,
            "raw_gzip_sha256": sha256_file(exp413_path),
            "decompressed_content_sha256": sha256_gzip_content(exp413_path),
            "visible_reference_decompressed_sha256": EXP413_VISIBLE_REFERENCE_CONTENT_SHA256,
            "matches_visible_reference": sha256_gzip_content(exp413_path)
            == EXP413_VISIBLE_REFERENCE_CONTENT_SHA256,
            "rows": int(exp413_runtime_metrics["rows"]),
            "wells": int(exp413_runtime_metrics["wells"]),
            "candidate_count": int(exp413_runtime_metrics["candidate_count"]),
            "parent_selector_model_count": int(
                exp413_runtime_metrics["parent_selector_model_count"]
            ),
            "signed_selector_model_count": int(
                exp413_runtime_metrics["signed_selector_model_count"]
            ),
            "tvt_model_count": int(exp413_runtime_metrics["tvt_model_count"]),
            "booster_training_count": int(
                exp413_runtime_metrics["booster_training_count"]
            ),
            "serialization_roundtrip_max_abs": exp413_serialization_roundtrip_max_abs,
            "runtime_seconds": float(exp413_runtime_metrics["runtime_seconds"]),
        },
        "rows": int(len(final)),
        "wells": int(component_table["well"].nunique()),
        "fallback_rows": 0,
        "duplicate_ids": int(final["id"].duplicated().sum()),
        "nonfinite_predictions": int((~np.isfinite(final["tvt"].to_numpy(np.float64))).sum()),
        "formula_parity_max_abs": {
            "public_preoverride": float(public_parity),
            "final": float(final_parity),
        },
        "seed_policy": "sha256(base_seed,split,family,well,seed_index)",
        "global_rng_calls": 0,
        "feature_audit": feature_audit,
        "component_prediction_content_sha256": {
            "sp45": dataframe_content_sha(sp45, ["id", "tvt"]),
            "pipeline_b": dataframe_content_sha(pipeline_b, ["id", "tvt"]),
            "public_preoverride": dataframe_content_sha(public_component, ["id", "tvt"]),
            "exp413": dataframe_content_sha(exp413, ["id", "tvt"]),
            "final": dataframe_content_sha(final, ["id", "tvt"]),
        },
        "generated_file_sha256": {
            "sp45_projection_submission.csv": sha256_file(sp45_path),
            "submission_B.csv": sha256_file(pipeline_b_path),
            "public_preoverride_submission.csv": sha256_file(public_path),
            "submission.csv": sha256_file(submission_path),
        },
        "runtime_seconds": float(time.time() - started),
        "deterministic_anchor": False,
        "deterministic_anchor_note": "requires a same-source/input rerun SHA match",
    }
    (CFG.OUT / "exp510_component_readout.json").write_text(
        json.dumps(readout, indent=2, sort_keys=True) + "\n"
    )
    (CFG.OUT / "exp510_reproducibility_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return final, manifest


if __name__ == "__main__":
    submission, reproducibility_manifest = run_inference()
    print(submission.head())
