# %% [markdown]
# # LATE SUBMIT — exp516 6th-place pfA × twGR faithful replay
#
# This is a post-competition reproduction audit, not an official-place submission.
# It replays only the published standalone pfA × typewell-GR component.

# %% [markdown]
# ## Contents
# 1. Imports and immutable source contract
# 2. Runtime, input, and checkpoint guards
# 3. Public GR-free anchor generator
# 4. Generate fold-safe GR-free hidden-test anchor
# 5. Public learned-emission helpers
# 6. Generate hidden-test similarity bands
# 7. Public GPU particle-filter and whole-smoother engine
# 8. Run pfA × twGR only
# 9. Sample-ID alignment, manifests, and LATE SUBMIT output

# %% [markdown]
# ## 1. Imports and immutable source contract
# The public notebook/config/checkpoint identities are frozen before any prediction. Reported author scores are external references, never outputs of this run.

# %%
from __future__ import annotations

import gzip
import hashlib
import json
import os
import pickle
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

EXPERIMENT = "exp516_sixth_place_pfa_tw_late_submit"
LATE_SUBMISSION_PHASE = "post_competition_late_submission"
PUBLIC_KERNEL = "k256net/public20th-private6th-pf-pf-pf-pf-and-bagging"
PUBLIC_KERNEL_ID_NO = 126919690
PUBLIC_NOTEBOOK_SHA256 = "b44f7889d6abdf9b027d33cb6c6b45f23902d609fae8d06f332914017784c924"
PUBLIC_CONFIG_SHA256 = "80e973d5f5e0e39be758a03f399cdd3d81d9e79320da8db6fbddbc25c2a202f3"
PUBLIC_CONFIG_TEXT_SHA256 = "aff2bcf63d1dd0b24ceefcd77a8b4fc058e0977e4e9518b58c6fea8c5468d962"
PUBLIC_ANCHOR_SOURCE_SHA256 = "7a93ca1f19636199a8fd942f32889bff6eb3d2f409488b47d5449877eafcb0f7"
PUBLIC_EMISSION_SOURCE_SHA256 = "d4592ba0279488036bdb01e12652d2066ee187861894ba985fa0730400268417"
ADAPTED_EMISSION_SOURCE_SHA256 = "01f0513b42f908b926131f1bb29fd451bc54daf7aad5aae9a42bd75e50aa3155"
PUBLIC_PF_SOURCE_SHA256 = "ea5e5af2a6fe6e344ad3a792c2735368dbe9f2f61aae79b804abe3eb493e6a6e"
PF_BANK = "pfA"
PF_REPRESENTATION = "tw"
PF_GENERATION_SEED = 4423098
PF_N_PARTICLES = 600
PF_N_SEEDS = 32
PF_SMOOTH_MODE = "full"
PF_WELL_CHUNK = 40
EXPECTED_CHECKPOINT_SHA256 = {'stageA_enccapaug_f0.pt': '9ce3763b14a68ae5f05e78467ae4faa5b696ab714c270fd129a6e7d468cdd007', 'stageA_enccapaug_f1.pt': 'd6b1dc0956e47b778bb2089c644d71c9006fdf8948a9cc962c6a40d27dccdfec', 'stageA_enccapaug_f2.pt': '2f312ce8de05feab9d8480a7971d91eb4e8289cce7e654f895f1b63d95cf5208', 'stageA_enccapaug_f3.pt': '8bf70f128c12a86e6f65177912d7ee90b5d24a7a67caf91f40df992bdd2c599f', 'stageA_enccapaug_f4.pt': 'b438d113b8fc07d8e9842cb66d01cfff2971ef4a5d213841574ec2680b2ab170'}
PUBLIC_CONFIG_JSON = '{\n "bank_order": [\n  "pf_1",\n  "pf_2",\n  "pf_3",\n  "r0_seed32",\n  "r1_seed32",\n  "pfA"\n ],\n "physics_banks": [\n  "pfA"\n ],\n "w_nn_bank": {\n  "pfA": 0.01\n },\n "w_nn_default": 0.02,\n "n_seed": 32,\n "smooth_lag": 192,\n "anchor_pkl": "grfree_anchor_train.pkl",\n "params": {\n  "pf_1": {\n   "gr_power": 2.1947763195255123,\n   "gr_sig_def": 114.11439615641156,\n   "gr_sig_max": 32.79346108797473,\n   "gr_sig_min": 7.495879205722816,\n   "gr_sig_mult": 2.7482077023565075,\n   "hgr_smooth_r": 3,\n   "init_pos_std": 0.43824435375436277,\n   "init_rate_std": 0.001840222365629066,\n   "jump_prob": 0.002,\n   "jump_std": 5.0,\n   "likelihood_scale": 20.0,\n   "mom": 0.998833416480357,\n   "n_particles": 600,\n   "pn": 0.006788762926604484,\n   "rate_clip": 2.0,\n   "resamp": 0.060642973044944336,\n   "rp": 0.6864944651997384,\n   "rr": 0.00015873788792956448,\n   "tvt_clip_margin": 113.08092552242186,\n   "tw_gr_smooth_r": 0,\n   "vn": 0.0013871256161405515,\n   "phys_sig": 0.05095402721954007,\n   "use_phys": false,\n   "smooth_lag": 192,\n   "use_anchor": false\n  },\n  "pf_2": {\n   "gr_sig_def": 71.37105679394953,\n   "gr_sig_max": 44.7373292550486,\n   "gr_sig_min": 6.520200142977573,\n   "gr_sig_mult": 1.9000654781073127,\n   "init_pos_std": 1.1039311766848976,\n   "init_rate_std": 0.009213697911348712,\n   "likelihood_scale": 18.584958539746772,\n   "mom": 0.9994265753132063,\n   "n_particles": 600,\n   "pn": 0.011257078147676399,\n   "resamp": 0.30834451379977984,\n   "rp": 0.49711567429512854,\n   "rr": 0.00031680690540158854,\n   "tvt_clip_margin": 63.752292532272826,\n   "vn": 0.0010650490363481595,\n   "rate_clip": 0.0,\n   "gr_power": 2.0,\n   "hgr_smooth_r": 0,\n   "tw_gr_smooth_r": 0,\n   "jump_prob": 0.0,\n   "jump_std": 10.0,\n   "phys_sig": 0.05095402721954007,\n   "use_phys": false,\n   "smooth_lag": 192,\n   "use_anchor": false\n  },\n  "pf_3": {\n   "gr_power": 2.074149546524975,\n   "gr_sig_def": 66.97913258656452,\n   "gr_sig_max": 50.843656140023995,\n   "gr_sig_min": 9.427442041004895,\n   "gr_sig_mult": 2.4485083695548218,\n   "hgr_smooth_r": 0,\n   "init_pos_std": 2.3279966476603975,\n   "init_rate_std": 0.03938758391370206,\n   "jump_prob": 0.002,\n   "jump_std": 5.0,\n   "likelihood_scale": 14.890876838217093,\n   "mom": 0.9999769474035675,\n   "n_particles": 600,\n   "pn": 0.04394815893193551,\n   "rate_clip": 0.0,\n   "resamp": 0.054606961294962585,\n   "rp": 0.42322457711970995,\n   "rr": 0.002148810696831248,\n   "tvt_clip_margin": 91.4794941498565,\n   "tw_gr_smooth_r": 0,\n   "vn": 0.0006883646035942229,\n   "phys_sig": 0.05095402721954007,\n   "use_phys": false,\n   "smooth_lag": 192,\n   "use_anchor": false,\n   "robust_nu": 5.0,\n   "temper_beta": 0.85\n  },\n  "r0_seed32": {\n   "n_particles": 400,\n   "likelihood_scale": 10.536319667317654,\n   "init_pos_std": 0.9331541486020234,\n   "init_rate_std": 0.00047753793163756304,\n   "mom": 0.9997911117425017,\n   "vn": 0.0006297672017716754,\n   "pn": 0.005014009140727263,\n   "rp": 0.7627309890952053,\n   "rr": 0.00045663430474523125,\n   "resamp": 0.032049849143995945,\n   "gr_sig_min": 11.865793233968713,\n   "gr_sig_max": 44.98927964612316,\n   "gr_sig_def": 119.25397464636981,\n   "gr_sig_mult": 3.058318309943017,\n   "tvt_clip_margin": 150.56111180234504,\n   "rate_clip": 2.0,\n   "gr_power": 1.7826896851395453,\n   "hgr_smooth_r": 0,\n   "tw_gr_smooth_r": 0,\n   "jump_prob": 0.002,\n   "jump_std": 5.0,\n   "phys_sig": 0.013666550145707291,\n   "use_phys": false,\n   "smooth_lag": 192,\n   "use_anchor": false,\n   "robust_nu": 5.0,\n   "temper_beta": 1.0\n  },\n  "r1_seed32": {\n   "n_particles": 400,\n   "likelihood_scale": 43.06423939744663,\n   "init_pos_std": 1.472927959941933,\n   "init_rate_std": 0.0006649111513262288,\n   "mom": 0.9995184091067627,\n   "vn": 0.0012160450940772792,\n   "pn": 0.002809741394025188,\n   "rp": 0.7070613341717283,\n   "rr": 0.0007812869001227403,\n   "resamp": 0.036600490411087865,\n   "gr_sig_min": 6.324062861615175,\n   "gr_sig_max": 29.32935945549601,\n   "gr_sig_def": 83.2600219362552,\n   "gr_sig_mult": 1.7565330745434495,\n   "tvt_clip_margin": 177.98411545472226,\n   "rate_clip": 2.5,\n   "gr_power": 1.4643521376101742,\n   "hgr_smooth_r": 0,\n   "tw_gr_smooth_r": 0,\n   "jump_prob": 0.002,\n   "jump_std": 10.0,\n   "phys_sig": 0.010953433688074062,\n   "use_phys": false,\n   "smooth_lag": 192,\n   "use_anchor": false,\n   "robust_nu": 5.0,\n   "temper_beta": 0.85\n  },\n  "pfA": {\n   "n_particles": 600,\n   "likelihood_scale": 29.056396339290437,\n   "init_pos_std": 1.6168048868046325,\n   "init_rate_std": 0.014412715870661067,\n   "mom": 0.9995682504305502,\n   "vn": 0.000873883338794689,\n   "pn": 0.00606209538025288,\n   "rp": 0.14025726060525087,\n   "rr": 0.00398567931602027,\n   "resamp": 0.18645022443117387,\n   "gr_sig_min": 10.63834771185878,\n   "gr_sig_max": 43.95858017409127,\n   "gr_sig_def": 86.87561008087175,\n   "gr_sig_mult": 2.7681905590609426,\n   "tvt_clip_margin": 86.30698497112195,\n   "rate_clip": 0.0,\n   "gr_power": 1.85175446220882,\n   "hgr_smooth_r": 2,\n   "tw_gr_smooth_r": 0,\n   "jump_prob": 0.002,\n   "jump_std": 5.0,\n   "phys_sig": 2.3072090423765363,\n   "use_anchor": true,\n   "anchor_mult": 20.0,\n   "anchor_ramp_md": 0.0,\n   "lk_floor": 1e-05,\n   "gr_debias": 0,\n   "rate_target": "zero",\n   "grid_step": 0.2,\n   "self_mix_w": 1.0,\n   "nbr_mix_w": 1.0,\n   "use_phys": false,\n   "smooth_lag": 192,\n   "robust_nu": 5.0,\n   "temper_beta": 1.0\n  }\n },\n "smooth_mode": "full"\n}'


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def logical_csv_sha(frame: pd.DataFrame) -> str:
    return sha256_text(frame.to_csv(index=False, lineterminator="\n"))

# %% [markdown]
# ## 2. Runtime, input, and checkpoint guards
# The current competition train/test and sample submission are discovered dynamically. The five public encoder files must match frozen SHA256 values; there is no emission-off fallback.

# %%
def resolve_competition_root() -> Path:
    roots = [
        Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction"),
        Path("/kaggle/input/rogii-wellbore-geology-prediction"),
    ]
    roots.extend(path.parent for path in Path("/kaggle/input").rglob("sample_submission.csv"))
    local = Path("data/raw").resolve()
    roots.append(local)
    for root in roots:
        if (root / "train").is_dir() and (root / "test").is_dir() and (root / "sample_submission.csv").is_file():
            return root
    raise FileNotFoundError("competition root with train/test/sample_submission.csv was not found")


def resolve_public_checkpoints(input_root: Path) -> dict[str, Path]:
    resolved: dict[str, Path] = {}
    for name, expected_sha in EXPECTED_CHECKPOINT_SHA256.items():
        exact = []
        for path in sorted(input_root.rglob(name)):
            if path.is_file() and sha256_path(path) == expected_sha:
                exact.append(path)
        if len(exact) != 1:
            raise RuntimeError(f"expected exactly one exact public checkpoint {name}, found {len(exact)}")
        resolved[name] = exact[0]
    return resolved


DATA_ROOT = resolve_competition_root()
SAMPLE_PATH = DATA_ROOT / "sample_submission.csv"
WORK_ROOT = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path(".").resolve()
ART_ROOT = WORK_ROOT / "exp516_public_pfa_artifacts"
NN50_ROOT = ART_ROOT / "nn50"
ART_ROOT.mkdir(parents=True, exist_ok=True)
NN50_ROOT.mkdir(parents=True, exist_ok=True)

if hashlib.sha256(PUBLIC_CONFIG_JSON.encode("utf-8")).hexdigest() != PUBLIC_CONFIG_TEXT_SHA256:
    raise RuntimeError("embedded public v96 config SHA drift")
config_path = ART_ROOT / "pf_banks_config.json"
config_path.write_text(PUBLIC_CONFIG_JSON, encoding="utf-8")

checkpoints = resolve_public_checkpoints(Path("/kaggle/input"))
for name, source in checkpoints.items():
    target = NN50_ROOT / name
    shutil.copy2(source, target)
    if sha256_path(target) != EXPECTED_CHECKPOINT_SHA256[name]:
        raise RuntimeError(f"checkpoint copy SHA mismatch: {name}")

gpu_count = torch.cuda.device_count()
gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
if gpu_count != 2 or not all("T4" in name for name in gpu_names):
    raise RuntimeError(f"official faithful run requires T4 x2, found {gpu_count}: {gpu_names}")

os.environ.update({
    "ROGII_DATA": str(DATA_ROOT),
    "ROGII_PROJ": str(WORK_ROOT),
    "ROGII_ART95": str(ART_ROOT),
    "ROGII_ART97": str(ART_ROOT),
    "PF_NGPU": "2",
    "PF_WELL_CHUNK": str(PF_WELL_CHUNK),
    "FULL_VRAM_GB": "8.0",
    "PYTHONUNBUFFERED": "1",
})

sample = pd.read_csv(SAMPLE_PATH)
if len(sample.columns) != 2 or list(sample.columns)[:1] != ["id"]:
    raise RuntimeError(f"sample submission must contain id plus one target, got {list(sample.columns)}")
TARGET_COLUMN = str(sample.columns[1])
if sample["id"].astype(str).duplicated().any():
    raise RuntimeError("sample submission contains duplicate ids")
print({
    "experiment": EXPERIMENT,
    "phase": LATE_SUBMISSION_PHASE,
    "data_root": str(DATA_ROOT),
    "sample_rows": int(len(sample)),
    "gpu_names": gpu_names,
    "public_kernel": PUBLIC_KERNEL,
})

# %% [markdown]
# ## 3. Public GR-free anchor generator
# This source is copied verbatim from the public submission notebook. It trains the fold-safe field/GRU anchor with five folds and three seeds; it is executed in a subprocess because the public script terminates with os._exit(0).

# %%
PUBLIC_ANCHOR_SOURCE = 'import io,sys,os,pickle\ntry: sys.stdout=io.open(1,"w",encoding="utf-8",closefd=False)\nexcept: pass\n# -*- coding: utf-8 -*-\n"""GRフリー自己完結 v2 — ★val完全除外の正しい5-fold。\n   fold毎に [場F・κ・近傍プール・GRU] の全てを train foldの井だけで構築 → val fold予測。\n   完全ランダムshuffle分割。fold別val CVを表示。GR一切不使用(X,Y,Z,TVT,TVT_inputのみ)。\n   test = 5fold×3seed=15セット(各foldのF/κ/GRU)平均。"""\nimport os, glob, time, io, sys\nimport numpy as np, pandas as pd\nt00 = time.time()\nKAGGLE = os.path.exists(\'/kaggle/input\')\ndef _resolve_root():\n    # ★呼び出し元(ノート/submit)が渡す ROGII_DATA を最優先\n    r = os.environ.get(\'ROGII_DATA\')\n    if r and os.path.isdir(os.path.join(r, \'train\')):\n        return r\n    for c in [\'/kaggle/input/rogii-wellbore-geology-prediction\',\n              \'/kaggle/input/competitions/rogii-wellbore-geology-prediction\']:\n        if os.path.isdir(os.path.join(c, \'train\')):\n            return c\n    g = glob.glob(\'/kaggle/input/*rogii*wellbore*\') or glob.glob(\'/kaggle/input/**/train\', recursive=True)\n    if g:\n        return g[0][:-6] if g[0].endswith(\'/train\') else g[0]\n    return r\'c:/Users/kosaka256/Documents/rogii_claude/rogii-wellbore-geology-prediction\'\nROOT = _resolve_root()\nTRAIN_DIR = os.path.join(ROOT, \'train\'); TEST_DIR = os.path.join(ROOT, \'test\'); SAMPLE = os.path.join(ROOT, \'sample_submission.csv\')\nPGATE = 0.35; THETA0 = 118.4; KNN_K, KNN_H = 15, 500.0\nKBINS = [0.0, 750.0, 1500.0, 2500.0, 4000.0, 1e18]; KAPPA_REGIMES = [0.0, 1000.0, 1500.0, 2000.0]\nNBIN = len(KBINS) - 1; CMAX = 0.30; K_FIX = 16; CLIP_DRIFT = 150.0\nN_FOLDS = 5; NBK = 3; STRIDE = 2; HID = 96; C = 37\nFT_EP = int(os.environ.get(\'FT_EP\', \'10\')); FT_LR = 1e-3; SEEDS = int(os.environ.get(\'SEEDS\', \'3\'))\nSPLIT_SEED = int(os.environ.get(\'SPLIT_SEED\', \'42\'))\nWINS = [301, 1001, 2001, 4001]; _C0, _S0 = np.cos(np.radians(THETA0)), np.sin(np.radians(THETA0))\ndef lowpass(x, win): return pd.Series(x).rolling(int(win) | 1, center=True, min_periods=1).mean().to_numpy()\n\n# ===== エンジン(phaseG lean, GR不使用) =====\ndef make_gdict(X, Y, Z, ti, tvt=None):\n    fin = np.where(np.isfinite(ti))[0]\n    if len(fin) == 0: return None\n    s = int(fin.max()); n = len(X) - 1 - s\n    if n < 2: return None\n    dX = np.diff(X)[s:]; dY = np.diff(Y)[s:]; Lxy = np.maximum(np.hypot(dX, dY), 1e-3)\n    p_row = (dX * _C0 + dY * _S0) / Lxy\n    d = dict(s=s, n=n, dX=dX, dY=dY, ndz=-np.diff(Z)[s:], Lxy=Lxy, arc=np.cumsum(Lxy),\n             Xl=X[s:], Yl=Y[s:], anchor=float(ti[s]), perp=np.abs(p_row) < PGATE)\n    if tvt is not None: d[\'R0\'] = tvt[s + 1:] - tvt[s]\n    return d\ndef segment_well(wd, with_slopes):\n    n, A = wd[\'n\'], wd[\'arc\']; K = min(K_FIX, max(1, n // 4)); total = float(A[-1])\n    edges = np.linspace(0.0, total, K + 1); segid = np.clip(np.searchsorted(edges[1:], A, side=\'left\'), 0, K - 1)\n    mid = np.empty((K, 2)); az = np.empty(K); Xl, Yl = wd[\'Xl\'], wd[\'Yl\']\n    for j in range(K):\n        rows = np.where(segid == j)[0]; j0, j1 = (int(rows[0]), int(rows[-1])) if len(rows) else (0, n - 1)\n        az[j] = np.arctan2(Yl[j1 + 1] - Yl[j0], Xl[j1 + 1] - Xl[j0]); mid[j] = ((Xl[j0] + Xl[j1 + 1]) / 2.0, (Yl[j0] + Yl[j1 + 1]) / 2.0)\n    if not with_slopes: return segid, mid, az, None, None\n    phi = np.column_stack([np.clip(A - edges[j], 0.0, edges[j + 1] - edges[j]) for j in range(K)])\n    y = wd[\'R0\'] - np.cumsum(wd[\'ndz\']); m = np.isfinite(y)\n    if m.sum() < K + 2: return segid, mid, az, None, None\n    c, res, rank, _ = np.linalg.lstsq(phi[m], y[m], rcond=None); dof = max(int(m.sum()) - K, 1)\n    ssr = float(res[0]) if len(res) else float(np.sum((phi[m] @ c - y[m]) ** 2))\n    cov = (ssr / dof) * np.linalg.pinv(phi[m].T @ phi[m]); se = np.sqrt(np.maximum(np.diag(cov), 1e-12))\n    return segid, mid, az, c, se\ndef query_well(F, mids, own, mdbuf=0.0):\n    keep = F[\'wi\'] != own; fx, fy = F[\'x\'][keep], F[\'y\'][keep]; fc = F[\'c\'][keep]; fp0 = F[\'p0\'][keep]; fqw = F[\'qw\'][keep]\n    dx = fx[None, :] - mids[:, 0:1]; dy = fy[None, :] - mids[:, 1:2]; d2 = dx * dx + dy * dy\n    if mdbuf > 0: d2 = np.where(d2 >= mdbuf * mdbuf, d2, np.inf)\n    kk = min(KNN_K, d2.shape[1] - 1) if d2.shape[1] > 1 else 1\n    idx = np.argpartition(d2, kk - 1, axis=1)[:, :kk]; r = np.arange(len(mids))[:, None]\n    d2s = d2[r, idx]; c_s, p0_s = fc[idx], fp0[idx]\n    w = np.exp(np.maximum(-d2s / (2 * KNN_H ** 2), -700.0)) * fqw[idx]; w[~np.isfinite(d2s)] = 0.0\n    dead = w.sum(1) <= 1e-300\n    with np.errstate(invalid=\'ignore\'): dd = np.sqrt(np.nanmedian(np.where(np.isfinite(d2s), d2s, np.nan), axis=1))\n    dd[~np.isfinite(dd)] = KBINS[-2] * 2\n    s_hat = (w * c_s * p0_s).sum(1) / ((w * p0_s ** 2).sum(1) + 1e-9); g = np.column_stack([s_hat * _C0, s_hat * _S0]); g[dead] = 0.0\n    gn = np.hypot(g[:, 0], g[:, 1]); big = gn > CMAX\n    if big.any(): g[big] *= (CMAX / gn[big])[:, None]\n    pred_don = g[:, 0:1] * np.cos(F[\'az\'][keep][idx]) + g[:, 1:2] * np.sin(F[\'az\'][keep][idx])\n    spread = np.sqrt(np.maximum((w * (c_s - pred_don) ** 2).sum(1) / np.maximum(w.sum(1), 1e-12), 0.0)); spread[dead] = 0.0\n    return g, dd, spread\ndef design_blocks(F, wd, mdbuf):\n    g, dd, _ = query_well(F, wd[\'seg\'][1], wd[\'wi\'], mdbuf=mdbuf); sid = wd[\'seg\'][0]\n    f = g[sid, 0] * wd[\'dX\'] + g[sid, 1] * wd[\'dY\']; a = wd[\'ndz\']; par = ~wd[\'perp\']\n    b_row = np.digitize(dd, KBINS[1:-1])[sid]; cols = []\n    for b in range(NBIN):\n        m = (b_row == b) & par; cols.append(np.cumsum(np.where(m, a, 0.0))); cols.append(np.cumsum(np.where(m, f, 0.0)))\n    cols.append(np.cumsum(np.where(wd[\'perp\'], a, 0.0))); cols.append(np.cumsum(np.where(wd[\'perp\'], f, 0.0)))\n    G = np.column_stack(cols); ok = np.isfinite(wd[\'R0\']); G = np.nan_to_num(G[ok]); return G.T @ G, G.T @ wd[\'R0\'][ok]\ndef solve_kappa(A, yv):\n    npar = 2 * NBIN; ncol = npar + 2; A = A.copy(); lam_p = 0.02 * float(np.trace(A)) / ncol\n    A[npar, npar] += lam_p; A[npar + 1, npar + 1] += lam_p; coef = np.linalg.lstsq(A, yv, rcond=None)[0]\n    return dict(alpha=np.clip(coef[0:npar:2], -0.25, 1.5), beta=np.clip(coef[1:npar:2], -0.25, 1.5),\n                ap=float(np.clip(coef[npar], -0.25, 1.5)), bp=float(np.clip(coef[npar + 1], -0.25, 1.5)))\ndef engine_predict(F, wd, K, own=-1):\n    g, dd, spread = query_well(F, wd[\'seg\'][1], own, mdbuf=0.0); sid = wd[\'seg\'][0]\n    f = g[sid, 0] * wd[\'dX\'] + g[sid, 1] * wd[\'dY\']; b_row = np.digitize(dd, KBINS[1:-1])[sid]\n    av = K[\'alpha\'][b_row]; bv = K[\'beta\'][b_row]; av = np.where(wd[\'perp\'], K[\'ap\'], av); bv = np.where(wd[\'perp\'], K[\'bp\'], bv)\n    rate = av * wd[\'ndz\'] + bv * f; resid = np.clip(np.cumsum(rate), -CLIP_DRIFT, CLIP_DRIFT)\n    return wd[\'anchor\'] + resid, rate, dd[sid], spread[sid]\n\ndef build_field(wells):\n    px, py, pc, paz, pwi = [], [], [], [], []\n    for wd in wells:\n        if wd[\'seg\'][3] is None: continue\n        segid, mid, az, c, se = wd[\'seg\']; ok = np.isfinite(c) & (np.abs(c) <= CMAX) & np.isfinite(se)\n        px.append(mid[ok, 0]); py.append(mid[ok, 1]); pc.append(c[ok]); paz.append(az[ok]); pwi.append(np.full(int(ok.sum()), wd[\'wi\'], float))\n    F = dict(x=np.concatenate(px), y=np.concatenate(py), c=np.concatenate(pc), az=np.concatenate(paz), wi=np.concatenate(pwi))\n    F[\'qw\'] = np.ones(len(F[\'x\'])); F[\'p0\'] = np.cos(F[\'az\']) * _C0 + np.sin(F[\'az\']) * _S0\n    return F\ndef fit_kappa(F, wells):\n    A = 0; yv = 0\n    for wd in wells:\n        if wd[\'seg\'][3] is None: continue\n        for R in KAPPA_REGIMES:\n            Ab, yb = design_blocks(F, wd, R); A = A + Ab; yv = yv + yb\n    return solve_kappa(A, yv)\ndef build_pool(wells):\n    ids = [wd[\'wid\'] for wd in wells]\n    cent = np.array([[np.nanmean(wd[\'_X\']), np.nanmean(wd[\'_Y\'])] for wd in wells])\n    from scipy.spatial import cKDTree\n    trees = {wd[\'wid\']: cKDTree(np.c_[wd[\'_X\'], wd[\'_Y\']][::2]) for wd in wells}\n    gks = {wd[\'wid\']: lowpass((wd[\'_tvt\'] + wd[\'_Z\']).astype(float), 201) for wd in wells}\n    mdc = {wd[\'wid\']: np.arange(len(wd[\'_X\']), dtype=float) for wd in wells}\n    yx = {wd[\'wid\']: (wd[\'_X\'].astype(float), wd[\'_Y\'].astype(float)) for wd in wells}\n    return dict(ids=ids, cent=cent, trees=trees, gks=gks, md=mdc, yx=yx)\n\ndef build_channels(md, x, y, z, tin, t0, d7_drift, rates_full, conf_full, POOL, self_wid=None):\n    n = len(md); ei = np.arange(t0 + 1, n)[::STRIDE].astype(int); tvt_last = float(tin[t0])\n    d7 = d7_drift[(ei - (t0 + 1))]; rates = rates_full[ei]; conf = conf_full[ei]\n    g = np.where(np.isfinite(tin), tin, np.nan) + z; ki = np.where(np.isfinite(tin))[0]\n    zh301 = z - lowpass(z, 301); tl = ki[-min(len(ki), 1000):]; gt = (g - lowpass(g, 301))[tl]; zt = zh301[tl]\n    fin = np.isfinite(gt); vz = float(np.dot(zt[fin], zt[fin])); k_pref = float(np.clip(np.dot(zt[fin], gt[fin]) / vz, 0, 1.5)) if vz > 1e-9 else 0.0\n    az = np.arctan2(np.gradient(y), np.gradient(x)); az_s = np.arctan2(lowpass(np.sin(az), 301), lowpass(np.cos(az), 301))\n    bands = {W: lowpass(z, W) for W in WINS}; msb = (np.arange(n) - t0).astype(float)\n    dh = np.hypot(np.gradient(lowpass(x, 101)), np.gradient(lowpass(y, 101))) + 1e-9\n    incl = np.arctan2(np.gradient(lowpass(z, 301)), dh); build_r = np.gradient(lowpass(incl, 201)) * 1000\n    g_s = lowpass(g, 201); r_all = np.gradient(np.nan_to_num(g_s, nan=0.0)) / np.maximum(dh, 1e-6)\n    tlp = ki[(ki >= t0 - 1500)]; rvals = r_all[tlp]; rvals = rvals[np.isfinite(g_s[tlp])]\n    r_pref = float(np.clip(np.median(rvals), -0.08, 0.08)) if len(rvals) > 100 else 0.0\n    dprior = (r_pref * (md - md[t0]) - (z - z[t0])); kn_flag = np.zeros(len(ei), np.float32)\n    ch = [d7 / 20.0, rates * 20.0, conf, kn_flag, msb[ei] / 3000.0, np.sin(az_s[ei]), np.cos(az_s[ei]),\n          np.full(len(ei), k_pref), incl[ei] * 3, build_r[ei], (incl[ei] - incl[t0]) * 3, np.clip(dprior[ei], -120, 120) / 30.0]\n    for W in WINS: ch.append((bands[W][ei] - bands[W][t0]) / 50.0); ch.append((z - bands[W])[ei] / 10.0)\n    cent = POOL[\'cent\']; ids = POOL[\'ids\']\n    dd0 = np.hypot(cent[:, 0] - np.nanmean(x), cent[:, 1] - np.nanmean(y)); order = np.argsort(dd0); nbrs = []\n    for j in order:\n        if self_wid is not None and ids[j] == self_wid: continue\n        nbrs.append(ids[j])\n        if len(nbrs) == NBK: break\n    p0 = np.array([x[t0], y[t0]]); g0 = float(tin[t0] + z[t0])\n    for nw in nbrs:\n        tr = POOL[\'trees\'][nw]; gks = POOL[\'gks\'][nw]; mdn = POOL[\'md\'][nw]; nx, ny = POOL[\'yx\'][nw]\n        _, j0 = tr.query(p0); j0 = int(j0) * 2\n        dq, jj = tr.query(np.c_[x[ei], y[ei]]); jj = (np.asarray(jj) * 2).astype(int)\n        dipk = np.gradient(gks, mdn, edge_order=1)\n        ch += [np.clip(gks[jj] - gks[j0], -120, 120) / 30.0, np.asarray(dq) / 1000.0, dipk[jj] * 30,\n               np.cos(az_s[ei] - np.arctan2(np.gradient(ny)[jj], np.gradient(nx)[jj])),\n               np.full(len(ei), float(np.clip(gks[j0] - g0, -120, 120)) / 30.0)]\n    ch += [np.zeros(len(ei), np.float32), np.zeros(len(ei), np.float32)]\n    X = np.nan_to_num(np.stack(ch, 1).astype(np.float32)); return X, ei, tvt_last\n\ndef wseq(gd, F, K, POOL, own):\n    X_, Y_, Z_, ti, tvt = gd[\'_X\'], gd[\'_Y\'], gd[\'_Z\'], gd[\'_ti\'], gd[\'_tvt\']\n    n = len(X_); d7_full, rate, ddr, spr = engine_predict(F, gd, K, own=own); t0w = gd[\'s\']\n    rates_full = np.zeros(n, np.float32); rates_full[t0w + 1:] = rate\n    conf_full = np.zeros(n, np.float32)\n    conf_full[t0w + 1:] = np.exp(-np.maximum(ddr - 800.0, 0.0) / 1000.0) * np.exp(-np.nan_to_num(spr, nan=0.1) / 0.06)\n    md = np.arange(n, dtype=float)\n    X, ei, tvt_last = build_channels(md, X_, Y_, Z_, ti, t0w, (d7_full - float(ti[t0w])), rates_full, conf_full, POOL, self_wid=gd[\'wid\'])\n    target = (tvt[ei] - d7_full[(ei - (t0w + 1))]).astype(np.float32)\n    return X, ei, target, d7_full, t0w\n\n# ===== torch / モデル定義(ロード経路でも必要なので先に定義) =====\nimport torch, torch.nn as nn\nfrom sklearn.model_selection import KFold\ndev = \'cuda\' if torch.cuda.is_available() else \'cpu\'\nclass Net(nn.Module):\n    def __init__(s):\n        super().__init__(); s.inp = nn.Linear(C + 1, HID); s.gru = nn.GRU(HID, HID, 2, batch_first=True, bidirectional=True, dropout=0.1); s.out = nn.Linear(HID * 2, 1)\n    def forward(s, x):\n        h = torch.relu(s.inp(x)); h, _ = s.gru(h); return s.out(h)[..., 0]\ndef train_loop(net, items, seed):\n    opt = torch.optim.AdamW(net.parameters(), lr=FT_LR, weight_decay=1e-4); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, FT_EP)\n    rng = np.random.default_rng(seed); BS = 12\n    for ep in range(FT_EP):\n        net.train(); order = rng.permutation(len(items))\n        for b0 in range(0, len(order), BS):\n            batch = [items[i] for i in order[b0:b0 + BS]]; L = max(len(b[1]) for b in batch)\n            xb = np.zeros((len(batch), L, C + 1), np.float32); yb = np.zeros((len(batch), L), np.float32); mb = np.zeros((len(batch), L), np.float32)\n            for i, (Xx, yv, fl) in enumerate(batch):\n                xb[i, :len(yv), :C] = Xx; xb[i, :len(yv), C] = fl; yb[i, :len(yv)] = yv; mb[i, :len(yv)] = 1.0\n            xb = torch.from_numpy(xb).to(dev); yb = torch.from_numpy(yb).to(dev); mb = torch.from_numpy(mb).to(dev)\n            loss = (nn.functional.huber_loss(net(xb), yb, delta=8.0, reduction=\'none\') * mb).sum() / mb.sum()\n            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step()\n        sch.step()\n    return net\n\ndef _rebuild_pool_trees(P):   # 保存時に外した cKDTree を yx から再構築(scipyバージョン非依存で再現一致)\n    from scipy.spatial import cKDTree\n    if P.get(\'trees\') is None:\n        P[\'trees\'] = {wid: cKDTree(np.c_[x, y][::2]) for wid, (x, y) in P[\'yx\'].items()}\n    return P\n\n# ===== 事前学習済み FOLD_ART があればロード → train全ロード+GRU学習を丸ごとスキップ =====\nFOLD_ART_PKL = os.environ.get(\'GRFREE_FOLD_ART\')   # (F_k,K_k,POOL_k,GRU重み)×5fold の保存/ロード先\n# ★FORCE_ANCHOR(最終錨pklの再生成)とは独立。FOLD_ARTは存在すれば常にロード(FORCE_FOLD_ART=1で強制再学習)\n_LOAD_FA = bool(FOLD_ART_PKL) and os.path.exists(FOLD_ART_PKL) and os.environ.get(\'FORCE_FOLD_ART\', \'0\') != \'1\'\nif _LOAD_FA:\n    blob = pickle.loads(open(FOLD_ART_PKL, \'rb\').read())\n    FOLD_ART = []\n    for (F_k, K_k, POOL_k, states) in blob:\n        _rebuild_pool_trees(POOL_k)\n        nets = []\n        for st in states:\n            net = Net().to(dev); net.load_state_dict({k: v.to(dev) for k, v in st.items()}); net.eval(); nets.append(net)\n        FOLD_ART.append((F_k, K_k, POOL_k, nets))\n    print(f\'[FOLD_ART] 事前学習済みロード {FOLD_ART_PKL} ({len(FOLD_ART)}fold) = train全ロード+GRU学習スキップ  ({time.time()-t00:.0f}s)\', flush=True)\nelse:\n    # ===== train 全ロード =====\n    wids = sorted(set(os.path.basename(f).split(\'__\')[0] for f in glob.glob(os.path.join(TRAIN_DIR, \'*__horizontal_well.csv\'))))\n    WELLS = []\n    for wi, w in enumerate(wids):\n        hw = pd.read_csv(os.path.join(TRAIN_DIR, f\'{w}__horizontal_well.csv\'), usecols=[\'X\', \'Y\', \'Z\', \'TVT\', \'TVT_input\'])\n        X = hw[\'X\'].to_numpy(float); Y = hw[\'Y\'].to_numpy(float); Z = hw[\'Z\'].to_numpy(float); tvt = hw[\'TVT\'].to_numpy(float); ti = hw[\'TVT_input\'].to_numpy(float)\n        gd = make_gdict(X, Y, Z, ti, tvt)\n        if gd is None: continue\n        gd[\'wi\'] = wi; gd[\'wid\'] = w; gd[\'seg\'] = segment_well(gd, with_slopes=True)\n        gd[\'_X\'] = X; gd[\'_Y\'] = Y; gd[\'_Z\'] = Z; gd[\'_ti\'] = ti; gd[\'_tvt\'] = tvt\n        WELLS.append(gd)\n    print(f\'train wells={len(WELLS)}  ({time.time()-t00:.0f}s)\', flush=True)\n\n    # ===== ★完全ランダムshuffle 5分割 → fold毎に[場・κ・プール・GRU] =====\n    idx = np.arange(len(WELLS))\n    FOLDS = list(KFold(N_FOLDS, shuffle=True, random_state=SPLIT_SEED).split(idx))   # ★完全ランダム\n    FOLD_ART = []   # (F_k, K_k, POOL_k, nets)\n    oof_e = []; oof_base = []; fold_cv = []\n    for fold, (tr_i, va_i) in enumerate(FOLDS):\n        tr_wells = [WELLS[i] for i in tr_i]; va_wells = [WELLS[i] for i in va_i]\n        # ★train foldの井だけで 場・κ・近傍プール(val完全除外)\n        F_k = build_field(tr_wells); K_k = fit_kappa(F_k, tr_wells); POOL_k = build_pool(tr_wells)\n        # train seq(own=wi で自井donor除外) → GRU学習items\n        items = []\n        for gd in tr_wells:\n            try:\n                X, ei, tg, d7, t0 = wseq(gd, F_k, K_k, POOL_k, own=gd[\'wi\'])\n                items.append((X, tg, 0.0)); items.append((X[::-1].copy(), (tg[::-1] - tg[-1]).copy(), 1.0))\n            except Exception: pass\n        nets = []\n        for sd in range(SEEDS):\n            torch.manual_seed(sd + 100 + fold); net = Net().to(dev); net = train_loop(net, items, sd + fold * 7); net.eval(); nets.append(net)\n        FOLD_ART.append((F_k, K_k, POOL_k, nets))\n        # val予測(val井は F_k/POOL_k に不在 → own=-1)\n        fe = []; fb = []\n        for gd in va_wells:\n            try:\n                X, ei, tg, d7, t0 = wseq(gd, F_k, K_k, POOL_k, own=-1)\n            except Exception: continue\n            xb = np.zeros((1, len(ei), C + 1), np.float32); xb[0, :, :C] = X\n            with torch.no_grad():\n                tb = torch.from_numpy(xb).to(dev); p = np.mean([nt(tb)[0].cpu().numpy() for nt in nets], axis=0)\n            tvt = gd[\'_tvt\']; pred = d7[(ei - (t0 + 1))] + p\n            fe.append(pred - tvt[ei]); fb.append(d7[(ei - (t0 + 1))] - tvt[ei])\n        fe = np.concatenate(fe); fb = np.concatenate(fb); oof_e.append(fe); oof_base.append(fb)\n        cvf = float(np.sqrt(np.mean(fe ** 2))); fold_cv.append(cvf)\n        print(f\'  fold{fold}: val井={len(va_wells)}  val CV(GRU)={cvf:.4f}  (engine単体 {float(np.sqrt(np.mean(fb**2))):.4f})  ({time.time()-t00:.0f}s)\', flush=True)\n    oof_e = np.concatenate(oof_e); oof_base = np.concatenate(oof_base)\n    CV = float(np.sqrt(np.mean(oof_e ** 2))); CVeng = float(np.sqrt(np.mean(oof_base ** 2)))\n    print(f\'\\n★★ OOF CV(GRフリー GRU, val完全除外) = {CV:.4f}  (engine単体 {CVeng:.4f})\', flush=True)\n    print(f\'   fold別 val CV = {[round(c,3) for c in fold_cv]}\', flush=True)\n    # ===== FOLD_ART 保存(GRU重み込み) → 次回以降はロードで学習スキップ =====\n    if FOLD_ART_PKL:\n        blob = []\n        for (F_k, K_k, POOL_k, nets) in FOLD_ART:\n            P2 = dict(POOL_k); P2[\'trees\'] = None      # cKDTreeは保存せずロード時にyxから再構築\n            states = [{k: v.detach().cpu() for k, v in nt.state_dict().items()} for nt in nets]\n            blob.append((F_k, K_k, P2, states))\n        with open(FOLD_ART_PKL, \'wb\') as fh: pickle.dump(blob, fh, protocol=4)\n        print(f\'[FOLD_ART] 保存 {FOLD_ART_PKL} ({len(blob)}fold, GRU重み込み)\', flush=True)\n\n# ===== test/対象SPLIT のアンカー生成(tvt=15本平均, sig=15本std) =====\nimport glob as _glob\nSPLIT=os.environ.get("GRF_SPLIT","test")\nDIR = TEST_DIR if SPLIT=="test" else TRAIN_DIR\nOUTPKL=os.environ.get("GRF_ANCHOR_OUT") or os.path.join(os.environ.get("ROGII_ART95", "."), f"grfree_anchor_{SPLIT}.pkl")\ntids=sorted(set(os.path.basename(f).split("__")[0] for f in _glob.glob(os.path.join(DIR,"*__horizontal_well.csv"))))\nANCH={}\nfor w in tids:\n    hw=pd.read_csv(os.path.join(DIR,f"{w}__horizontal_well.csv"),usecols=["X","Y","Z","TVT_input"])\n    X_=hw["X"].to_numpy(float);Y_=hw["Y"].to_numpy(float);Z_=hw["Z"].to_numpy(float);ti=hw["TVT_input"].to_numpy(float);n=len(X_)\n    finr=np.where(np.isfinite(ti))[0]\n    if len(finr)==0: continue\n    gd=make_gdict(X_,Y_,Z_,ti)\n    if gd is None: continue\n    gd["seg"]=segment_well(gd,with_slopes=False);gd["wid"]=w;t0w=gd["s"];tvt_last=float(ti[t0w]);md=np.arange(n,dtype=float)\n    seedpreds=[]\n    for (F_k,K_k,POOL_k,nets) in FOLD_ART:\n        d7_full,rate,ddr,spr=engine_predict(F_k,gd,K_k,own=-1)\n        rates_full=np.zeros(n,np.float32);rates_full[t0w+1:]=rate\n        conf_full=np.zeros(n,np.float32);conf_full[t0w+1:]=np.exp(-np.maximum(ddr-800.0,0.0)/1000.0)*np.exp(-np.nan_to_num(spr,nan=0.1)/0.06)\n        Xc,ei,_=build_channels(md,X_,Y_,Z_,ti,t0w,(d7_full-tvt_last),rates_full,conf_full,POOL_k,self_wid=None)\n        xb=np.zeros((1,len(ei),C+1),np.float32);xb[0,:,:C]=Xc\n        with torch.no_grad():\n            tb=torch.from_numpy(xb).to(dev)\n            for nt in nets:\n                pp=nt(tb)[0].cpu().numpy()\n                seedpreds.append(np.interp(np.arange(t0w+1,n),ei,d7_full[(ei-(t0w+1))]+pp))  # 全eval行TVT予測(1net)\n    if not seedpreds: continue\n    S=np.stack(seedpreds,0)                                # (15, eval行)\n    ANCH[w]=dict(tvt=S.mean(0).astype(np.float32), sig=np.clip(S.std(0),0.5,60.0).astype(np.float32))\npickle.dump(ANCH,open(OUTPKL,"wb"))\nprint(f"saved {OUTPKL}  {len(ANCH)}井 (GRフリー {SPLIT} アンカー tvt/sig)",flush=True)\nsys.stdout.flush()\nos._exit(0)   # ★torch/CUDA 終了時クラッシュ(Windows STATUS_STACK_BUFFER_OVERRUN)回避=保存後に即クリーン終了\n'

# %% [markdown]
# ## 4. Generate fold-safe GR-free hidden-test anchor
# The saved fold artifact contains the five field/pool states and 15 GRU state dicts. The current hidden test anchor is regenerated from runtime inputs; no visible-test anchor is reused.

# %%
anchor_script = WORK_ROOT / "exp516_public_gen_grfree_anchor.py"
anchor_script.write_text(PUBLIC_ANCHOR_SOURCE, encoding="utf-8")
if sha256_path(anchor_script) != PUBLIC_ANCHOR_SOURCE_SHA256:
    raise RuntimeError("public anchor source SHA mismatch after materialization")

anchor_path = ART_ROOT / "grfree_anchor_test.pkl"
fold_artifact_path = ART_ROOT / "grfree_fold_art.pkl"
anchor_env = dict(os.environ)
anchor_env.update({
    "GRF_SPLIT": "test",
    "GRF_ANCHOR_OUT": str(anchor_path),
    "GRFREE_FOLD_ART": str(fold_artifact_path),
    "FORCE_FOLD_ART": "1",
    "FT_EP": "10",
    "SEEDS": "3",
    "SPLIT_SEED": "42",
})
anchor_started = time.perf_counter()
subprocess.run([sys.executable, str(anchor_script)], env=anchor_env, check=True)
anchor_seconds = time.perf_counter() - anchor_started
if not anchor_path.is_file() or not fold_artifact_path.is_file():
    raise RuntimeError("public anchor generator did not produce both anchor and fold artifact")
anchor_payload = pickle.loads(anchor_path.read_bytes())
os.environ["NN_SPLIT"] = "test"
os.environ["V93_ANCHOR_PKL"] = str(anchor_path)
print({
    "anchor_wells": len(anchor_payload),
    "anchor_seconds": anchor_seconds,
    "anchor_sha256": sha256_path(anchor_path),
    "fold_artifact_sha256": sha256_path(fold_artifact_path),
})

# %% [markdown]
# ## 5. Public learned-emission helpers
# The encoder architecture and similarity-band construction are the public implementation. Only the hard-coded Windows project/data resolver is replaced by ROGII_PROJ/ROGII_DATA for Kaggle portability.

# %%
# -*- coding: utf-8 -*-
"""v93 module2: NN-emission sim 生成(★帯中心=GRフリー錨。過去struct不使用)。

v50/v51 の学習emission(TCN cosine類似度)を、候補帯の中心だけ struct→GRフリー錨tvt に差替。
  - encoder(stageA_enccapaug_f{f}.pt)は凍結再利用(v93/artifacts_v93/nn50 に複製済)。
  - OOF fold: sorted train井の strided 5-fold(wells[f::5])。fold f の井は fold f を hold-out した
    encoder f を使う=OOF-safe(v50/v51と同一分割)。Fh/Ft は encoder が期待する入力なので
    v50 cacheA と数値一致を assert してから使用。
  - 出力: v93/artifacts_v93/sim_grfree_v97.pkl = {wid: {sim(T,181)fp16, st(T,)=GRフリー錨tvt}}。
create 側は各PF入力dictに _sim/_st を添付して smoother に渡す(module1 _pad が消費)。
"""
import os, sys, io, glob, pickle
try:
    sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
except Exception:
    pass
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJ = Path(os.environ.get("ROGII_PROJ", "."))
DATA_DIR = Path(os.environ.get("ROGII_DATA", str(PROJ / "rogii-wellbore-geology-prediction")))
ART = Path(os.environ.get("ROGII_ART97", str(PROJ / "v97" / "artifacts_v97")))
NN50 = ART / "nn50"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SIM_STEP = 0.5
NBAND = 90                        # ±45ft / 0.5ft

# ---- split 対応(env NN_SPLIT) ----
#   train: 錨=grfree_anchor_train.pkl / OOF fold(wells[f::5]) の encoder f を使用。
#   test : 錨=env V93_ANCHOR_PKL(=GRフリー test 錨, module1 と同一 pkl)。
#          test 井はどの学習fold にも入っていないため encoder は 5本を平均する(OOF ではなく全平均)。
NN_SPLIT = os.environ.get("NN_SPLIT", "train")
if NN_SPLIT == "test":
    _ANCHOR_PKL = Path(os.environ["V93_ANCHOR_PKL"])   # test 錨(必須)
    SIM_FP = ART / "sim_grfree_test_v97.pkl"
else:
    _ANCHOR_PKL = ART / "grfree_anchor_train.pkl"
    SIM_FP = ART / "sim_grfree_v97.pkl"
_ANCHOR = pickle.loads(_ANCHOR_PKL.read_bytes())        # {wid:{tvt,sig}}


# ---- GR特徴(v50/v52 と同一。encoder入力) ----
def _rz(v):
    med = np.nanmedian(v); iqr = np.nanpercentile(v, 75) - np.nanpercentile(v, 25)
    return np.clip((v - med) / max(iqr / 1.349, 1e-6), -8, 8)


def _rm(v, w):
    return pd.Series(v).rolling(w, center=True, min_periods=1).mean().to_numpy()


def _rs(v, w):
    return pd.Series(v).rolling(w, center=True, min_periods=2).std().fillna(0).to_numpy()


def _detr(v, w=301):
    m = _rm(v, w); sd = _rs(v, w)
    return np.clip((v - m) / np.maximum(sd, 1e-3), -6, 6)


def feats_h(gr, md, z):
    g = pd.Series(gr).interpolate(limit_direction="both").to_numpy()
    g = np.where(np.isfinite(g), g, np.nanmedian(g)); g = np.where(np.isfinite(g), g, 0.0)
    z0 = _rz(g); s5, s21, s81 = _rm(z0, 5), _rm(z0, 21), _rm(z0, 81)
    d5 = np.gradient(s5) * 10.0; v21 = _rs(z0, 21); dt = _detr(z0, 301)
    dmd = np.maximum(np.gradient(md), 1e-3)
    dzdm = np.clip(np.gradient(z) / dmd, -0.2, 0.2) * 5.0
    X = np.stack([z0, s5, s21, s81, d5, v21, dt, dzdm], 1).astype(np.float32)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def feats_t(gg):
    z0 = _rz(gg); s3, s9, s33 = _rm(z0, 3), _rm(z0, 9), _rm(z0, 33)
    d3 = np.gradient(s3) * 10.0; v9 = _rs(z0, 9); dt = _detr(z0, 301)
    X = np.stack([z0, s3, s9, s33, d3, v9, dt], 1).astype(np.float32)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


# ---- encoder(capaug: ch=128, dils 6段。凍結ckpt に一致) ----
class Enc1D(nn.Module):
    def __init__(self, cin, ch=128, dout=96, dils=(1, 2, 4, 8, 16, 32)):
        super().__init__()
        self.inp = nn.Conv1d(cin, ch, 5, padding=2)
        self.blocks = nn.ModuleList()
        for d in dils:
            self.blocks.append(nn.Sequential(
                nn.Conv1d(ch, ch, 5, padding=2 * d, dilation=d), nn.GELU(), nn.Conv1d(ch, ch, 1)))
        self.norm = nn.ModuleList([nn.GroupNorm(8, ch) for _ in dils])
        self.out = nn.Conv1d(ch, dout, 1)

    def forward(self, x):
        h = self.inp(x.t()[None])
        for blk, nm in zip(self.blocks, self.norm):
            h = nm(h + blk(h))
        return F.normalize(self.out(h)[0].t(), dim=1)


def _load_encoders():
    encs = []
    for f in range(5):
        ck = torch.load(NN50 / f"stageA_enccapaug_f{f}.pt", map_location=DEVICE)
        eA = Enc1D(8).to(DEVICE); eA.load_state_dict(ck["encA"]); eA.eval()
        eB = Enc1D(7).to(DEVICE); eB.load_state_dict(ck["encB"]); eB.eval()
        encs.append((eA, eB))
    return encs


def _well_gg(tw):
    """typewell を 0.5ft TVT格子へ(sim帯の座標系)。返り (gg, gmin) or None。"""
    tt = tw["TVT"].to_numpy(float); tg = tw["GR"].to_numpy(float)
    m = np.isfinite(tt) & np.isfinite(tg); tt, tg = tt[m], tg[m]
    if len(tt) < 8:
        return None
    o = np.argsort(tt); tt, tg = tt[o], tg[o]
    gmin = float(tt.min())
    gg = np.interp(np.arange(gmin, float(tt.max()) + SIM_STEP, SIM_STEP), tt, tg)
    return gg, gmin


def build_sims(verify_cache=True, split=None):
    """split井の sim を GRフリー錨帯中心で生成。
       train: OOF fold=wells[f::5](井index i の fold=i%5 の encoder f)。
       test : どの学習fold にも入らないため encoder 5本を平均(全平均)。錨=_ANCHOR(=test錨)。"""
    split = split or NN_SPLIT
    is_test = (split == "test")
    TR = DATA_DIR / split
    wells = sorted({os.path.basename(q).split("__")[0] for q in glob.glob(str(TR / "*__horizontal_well.csv"))})
    encs = _load_encoders()
    offs = np.arange(-NBAND, NBAND + 1)
    # 検証用 cacheA(Fh/Ft の厳密一致確認, train のみ)。無ければskip
    cache = None
    if verify_cache and not is_test:
        cp = PROJ / "v50" / "artifacts_v50" / "cacheA_v50.pkl"
        if cp.exists():
            cache = pickle.loads(cp.read_bytes())
    out = {}; nver = 0; maxdev = 0.0
    print(f"[NN-emission] split={split} wells={len(wells)} 錨={_ANCHOR_PKL.name}({len(_ANCHOR)}井) "
          f"encoder={'全5平均' if is_test else 'OOF fold'} -> {SIM_FP.name}", flush=True)
    with torch.no_grad():
        for i, wn in enumerate(wells):
            enc_use = encs if is_test else [encs[i % 5]]   # test=全5, train=fold i%5 のみ
            sp = _ANCHOR.get(wn)
            hw = pd.read_csv(TR / f"{wn}__horizontal_well.csv")
            tw = pd.read_csv(TR / f"{wn}__typewell.csv")
            evm = hw["TVT_input"].isna().to_numpy()
            if sp is None or len(sp["tvt"]) != int(evm.sum()):
                continue                                # 錨欠損/行数不一致 → sim無し(注入OFF)
            gg_gmin = _well_gg(tw)
            if gg_gmin is None:
                continue
            gg, gmin = gg_gmin; K = len(gg)
            Fh = feats_h(hw["GR"].to_numpy(float), hw["MD"].to_numpy(float), hw["Z"].to_numpy(float))
            Ft = feats_t(gg)
            if cache is not None and wn in cache and nver < 30:
                cFh = np.asarray(cache[wn]["Fh"]); cFt = np.asarray(cache[wn]["Ft"])
                if cFh.shape == Fh.shape and cFt.shape == Ft.shape:
                    dev = max(float(np.abs(cFh - Fh).max()), float(np.abs(cFt - Ft).max()))
                    maxdev = max(maxdev, dev); nver += 1
            ev_idx = np.where(evm)[0]
            st = np.asarray(sp["tvt"], np.float64)      # ★帯中心=GRフリー錨tvt(struct不使用)
            ks = np.round((st - gmin) / SIM_STEP).astype(np.int64)
            cand = np.clip(ks[:, None] + offs[None, :], 0, K - 1)
            okc = (ks[:, None] + offs[None, :] >= 0) & (ks[:, None] + offs[None, :] <= K - 1)
            xh = torch.from_numpy(Fh).to(DEVICE); xt = torch.from_numpy(Ft).to(DEVICE)
            ct = torch.tensor(cand, device=DEVICE)
            sims_acc = None
            for (eA, eB) in enc_use:                     # test は 5本の類似度を平均
                a = eA(xh)[ev_idx]; b = eB(xt)
                s_e = (a[:, None, :] * b[ct]).sum(-1)
                sims_acc = s_e if sims_acc is None else (sims_acc + s_e)
            sims = (sims_acc / len(enc_use)).cpu().numpy()
            sims = np.where(okc, sims, -1.0)
            out[wn] = dict(sim=sims.astype(np.float16), st=st.astype(np.float32))
            if (i + 1) % 100 == 0:
                print(f"  sim {i+1}/{len(wells)} (ready {len(out)})", flush=True)
    if nver:
        print(f"[検証] Fh/Ft vs cacheA 最大乖離={maxdev:.2e} ({nver}井) -> 0付近ならencoder入力一致OK")
    SIM_FP.write_bytes(pickle.dumps(out, protocol=4))
    print(f"sim saved: {len(out)}/{len(wells)} wells -> {SIM_FP.name}")
    return out

# %% [markdown]
# ## 6. Generate hidden-test similarity bands
# All five frozen public encoders are averaged for test wells around the GR-free anchor at 0.5-ft spacing over ±45 ft.

# %%
similarity_started = time.perf_counter()
SIMD = build_sims(verify_cache=False, split="test")
similarity_seconds = time.perf_counter() - similarity_started
similarity_path = ART_ROOT / "sim_grfree_test_v97.pkl"
if not similarity_path.is_file():
    raise RuntimeError("public learned-emission builder did not save similarity artifact")
if len(SIMD) != len(anchor_payload):
    missing = sorted(set(anchor_payload) - set(SIMD))
    raise RuntimeError(f"learned emission missing anchor wells: {missing[:10]}")
print({
    "similarity_wells": len(SIMD),
    "similarity_seconds": similarity_seconds,
    "similarity_sha256": sha256_path(similarity_path),
})

# %% [markdown]
# ## 7. Public GPU particle-filter and whole-smoother engine
# This is the public pf_banks_v95 numerical engine with only its terminal standalone smoke block removed. The public v96 config and freshly generated anchor are loaded from ART_ROOT.

# %%
# -*- coding: utf-8 -*-
"""v93 module1: 6バンク平滑PFエンジン(GPU固定ラグ smoother)。

v52 create の smooth-PF 部分をクリーン再構築。★v93 の唯一の実変更:
  - pfA バンクの錨を「過去struct(v38/v66)」から「GRフリー錨(v91 OOF, sig=9ft較正)」へ差替。
  - 錨強度 anchor_mult は global 定数でなく バンクparam P["anchor_mult"] から取る(pfA=20)。
過去struct は一切読まない([[no-struct-directive]])。錨源は v93/artifacts_v93 のみ。

構成(v52 と数値一致させる要素):
  - build_smoother_inputs / build_inputs_self / build_inputs_nbr : PF入力(tw/self/nbr疑似typewell)
  - attach_anchor : GRフリー錨(anc/ancs)を添付。非physicsバンクは ancs=1e9(錨OFF)
  - _smoother_core : 固定ラグ(L=smooth_lag)平滑PF。GR尤度×phys×錨×NN-emission
  - run_smoother_ext : seed尤度加重で smoothed平均/std を返す
NN-emission(sim/帯中心 st)は create 側が入力dictに _sim/_st を添付する形で受ける(module2が生成)。
"""
import os, json, pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch

PROJ = Path(os.environ.get("ROGII_PROJ", r"C:\Users\kosaka256\Documents\rogii_claude"))
DATA_DIR = Path(os.environ.get("ROGII_DATA", str(PROJ / "rogii-wellbore-geology-prediction")))
ART = Path(os.environ.get("ROGII_ART95", str(PROJ / "v95" / "artifacts_v95")))  # ★v95: config/錨をv93からコピー済(PF param同一)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32

# ---- config(6バンクparam) & GRフリー錨 ----
_CFG = json.loads((ART / "pf_banks_config.json").read_text(encoding="utf-8"))
BANK_ORDER = _CFG["bank_order"]                    # [pf_1,pf_2,pf_3,r0_seed32,r1_seed32,pfA]
PHYSICS_BANKS = set(_CFG["physics_banks"])         # {pfA}
BANK_PARAMS = _CFG["params"]                       # dict[bank] -> raw param dict
W_NN_BANK = _CFG["w_nn_bank"]                       # {pfA:0.01}
W_NN_DEFAULT = float(_CFG["w_nn_default"])          # 0.02
SMOOTH_LAG = int(_CFG["smooth_lag"])               # 16
SMOOTH_MODE_CFG = str(_CFG.get("smooth_mode", "fixedlag"))   # ★v99: full平滑切替
N_SEED = int(_CFG["n_seed"])                        # 32
# ★v102: ps_combo(seed集約を尤度×錨カーネルに=選択強化)。env で全体上書き可、off_banksは無効。
PS_COMBO_TAU = float(os.environ.get("PS_COMBO_TAU", _CFG.get("ps_combo_tau", 0.0)))
PS_COMBO_OFF = set(_CFG.get("ps_combo_off_banks", []))

_ANCHOR_PKL = ART / _CFG.get("anchor_pkl", "grfree_anchor_train.pkl")
if os.environ.get("V93_ANCHOR_PKL"):
    _ANCHOR_PKL = Path(os.environ["V93_ANCHOR_PKL"])
_ANCHOR = pickle.loads(_ANCHOR_PKL.read_bytes())   # {wid: {tvt, sig}}
print(f"[v93 pf_banks] 錨={_ANCHOR_PKL.name} {len(_ANCHOR)}井 / banks={BANK_ORDER} / smooth_lag={SMOOTH_LAG}")

# ---- NN-emission 帯(create/module2と共有する幾何定数) ----
SIM_STEP_NN = 0.5
NBAND_NN = 90                                       # ±45ft / 0.5ft = 90

# ---- self/nbr/Z傾斜 定数(v52同値) ----
SELF_MIX_W = 1.0
NBR_MIX_W = 1.0
ZGRAD_R = 25
K_MAX = 3
MAX_DIST = 1500.0
NEED_REFS = 2


def bank_param(bank):
    """バンク生paramに physics/w_nn を付けて返す(create が smoother に渡す P)。"""
    P = dict(BANK_PARAMS[bank])
    P["name"] = bank
    P.setdefault("smooth_lag", SMOOTH_LAG)
    P.setdefault("smooth_mode", SMOOTH_MODE_CFG)
    P["_physics"] = bank in PHYSICS_BANKS
    P["_w_nn"] = float(W_NN_BANK.get(bank, W_NN_DEFAULT))
    P["_ps_combo_tau"] = 0.0 if bank in PS_COMBO_OFF else PS_COMBO_TAU   # ★v102 選択強化(off_banksは無効)
    return P


def _ps_combo_reweight(ww, ps_jT, st, tau):
    """★v102 ps_combo: seed尤度重み ww を「錨に近いseedを重く」再加重(選択強化)。st無/tau0はそのまま。
       ps_jT=(S,T) その井の per-seed平滑軌跡, st=GRフリー錨tvt(=_st, emission帯中心)。"""
    if tau <= 0 or st is None:
        return ww
    T = ps_jT.shape[1]; stj = np.asarray(st, float)
    if len(stj) < T:
        return ww
    da = ((ps_jT - stj[:T][None, :]) ** 2).mean(1)                      # 各seedの錨距離²
    w2 = ww * np.exp(-(da - da.min()) / (2.0 * tau * tau)); s = w2.sum()
    return w2 / s if s > 1e-300 else np.full(len(ww), 1.0 / len(ww))


# ==================== 小物(v52 と同一) ====================
def _smooth_radius_values(vals, fb, r):
    r = int(r)
    s = pd.Series(vals, dtype="float32").interpolate(limit_direction="both").fillna(float(fb))
    if r <= 0:
        return s.to_numpy(np.float32)
    return s.rolling(2 * r + 1, center=True, min_periods=1).mean().to_numpy(np.float32)


def _grid(tw_tvt, tw_gr, step=0.2):
    tmin = float(tw_tvt.min()); tmax = float(tw_tvt.max())
    tvt_g = np.arange(tmin, tmax + step, step)
    return np.interp(tvt_g, tw_tvt, tw_gr).astype(np.float64), float(tmin), float(step)


def _bin_grid(t, g, gmin, step, G):
    """(TVT, GR) を grid の TVT ビンに median 集約(疑似typewell)。無データビン=NaN。"""
    m = np.isfinite(t) & np.isfinite(g); t, g = t[m], g[m]
    idx = np.round((t - gmin) / step).astype(int); ok = (idx >= 0) & (idx < G); idx, gg = idx[ok], g[ok]
    out = np.full(G, np.nan)
    if len(gg):
        s = pd.Series(gg).groupby(idx).median(); out[s.index.to_numpy()] = s.to_numpy()
    return out


# ==================== PF入力(tw / self / nbr) ====================
def build_smoother_inputs(hw, tw_tvt, tw_gr, P):
    """smooth PF 入力を hw/tw と生PFパラメータから作る。eval無しは None。"""
    kn = hw[hw["TVT_input"].notna()]; ev = hw[hw["TVT_input"].isna()]
    if len(ev) == 0:
        return None
    tw_fb = float(np.nanmean(tw_gr)) if np.isfinite(np.nanmean(tw_gr)) else 0.0
    tw_gr_pf = _smooth_radius_values(tw_gr, tw_fb, P["tw_gr_smooth_r"]).astype(np.float64)
    gg, gmin, gst = _grid(tw_tvt, tw_gr_pf)
    gr_full = hw["GR"].astype(float).interpolate(limit_direction="both").fillna(tw_fb).to_numpy(float)
    gr_sm = _smooth_radius_values(gr_full, tw_fb, P["hgr_smooth_r"])
    ev_pos = hw.index.get_indexer(ev.index); kpos = hw.index.get_indexer(kn.index)
    kn_tvtin = kn["TVT_input"].to_numpy(float)
    if len(kpos) < 20:
        gs = P["gr_sig_def"]
    else:
        resid = gr_sm[kpos] - np.interp(kn_tvtin, tw_tvt, tw_gr_pf)
        gs = float(np.nanstd(resid)); gs = P["gr_sig_def"] if (not np.isfinite(gs) or gs <= 0) else gs
    gs = float(np.clip(gs * P["gr_sig_mult"], P["gr_sig_min"], max(P["gr_sig_max"], P["gr_sig_min"] + 1e-6)))
    gs = gs * float(os.environ.get("GS_SCALE", "1.0"))     # ★post-clip の GR尤度σ 広げ(gs×1.30 検証。既定1.0=無効)
    ls = float(kn["TVT_input"].iloc[-1] + kn["Z"].iloc[-1])
    tail = kn.tail(30); dt = np.diff(tail["TVT_input"].values); dz = np.diff(tail["Z"].values)
    dm = np.diff(tail["MD"].values); m = dm > 0
    ir = float(np.median((dt + dz)[m] / dm[m])) if m.sum() >= 3 else 0.0
    return dict(md=ev["MD"].to_numpy(float), z=ev["Z"].to_numpy(float), gr=gr_sm[ev_pos].astype(np.float64),
                gg=gg, gmin=gmin, gst=gst, gs=gs, ls=ls, ir=ir, N=int(P["n_particles"]))


def build_inputs_self(hw, tw_tvt, tw_gr, P):
    """tw入力の gg を、自分の既知prefix GR(TVT_input) 疑似typewell で上書き。"""
    base = build_smoother_inputs(hw, tw_tvt, tw_gr, P)
    if base is None:
        return None
    kn = hw[hw["TVT_input"].notna()]
    tw_fb = float(np.nanmean(tw_gr)) if np.isfinite(np.nanmean(tw_gr)) else 0.0
    gr_full = hw["GR"].astype(float).interpolate(limit_direction="both").fillna(tw_fb).to_numpy(float)
    gr_sm = _smooth_radius_values(gr_full, tw_fb, P["hgr_smooth_r"])
    kpos = hw.index.get_indexer(kn.index)
    gh = _bin_grid(kn["TVT_input"].to_numpy(float), gr_sm[kpos].astype(float), base["gmin"], base["gst"], len(base["gg"]))
    gg = base["gg"].copy(); have = np.isfinite(gh)
    gg[have] = SELF_MIX_W * gh[have] + (1 - SELF_MIX_W) * gg[have]
    base = dict(base); base["gg"] = gg
    return base


def build_inputs_nbr(hw, tw_tvt, tw_gr, P, refs):
    """tw入力の gg を、近傍train坑井の GR(TVT) 疑似typewell で上書き。refs無し=tw fallback。
       refs = [(gr_array, tvt_array), ...](module3 の近傍選択が供給)。"""
    base = build_smoother_inputs(hw, tw_tvt, tw_gr, P)
    if base is None:
        return None
    if refs:
        rt = np.concatenate([r[1] for r in refs])
        rg = np.concatenate([_smooth_radius_values(r[0], float(np.nanmean(r[0])), P["hgr_smooth_r"]).astype(float) for r in refs])
        gh = _bin_grid(rt, rg, base["gmin"], base["gst"], len(base["gg"]))
        gg = base["gg"].copy(); have = np.isfinite(gh)
        gg[have] = NBR_MIX_W * gh[have] + (1 - NBR_MIX_W) * gg[have]
        base = dict(base); base["gg"] = gg
    return base


def build_inputs_self_graft(hw, tw_tvt, tw_gr, P):
    """★v95新規: self(自分のprefix GR)を tw格子に substitute し、prefix範囲外(データ無しビン)を
       tw を self較正(self≈a·tw+b, a∈[0.2,5])で外挿。self被覆を広げる(v49 build_inputs_self_ext 由来)。
       較正係数 a,b・有効フラグ・外挿割合を base['_graft'] にメタ格納(provenance用)。"""
    base = build_smoother_inputs(hw, tw_tvt, tw_gr, P)
    if base is None:
        return None
    kn = hw[hw["TVT_input"].notna()]
    tw_fb = float(np.nanmean(tw_gr)) if np.isfinite(np.nanmean(tw_gr)) else 0.0
    gr_full = hw["GR"].astype(float).interpolate(limit_direction="both").fillna(tw_fb).to_numpy(float)
    gr_sm = _smooth_radius_values(gr_full, tw_fb, P["hgr_smooth_r"])
    kpos = hw.index.get_indexer(kn.index)
    gh = _bin_grid(kn["TVT_input"].to_numpy(float), gr_sm[kpos].astype(float), base["gmin"], base["gst"], len(base["gg"]))
    gg = base["gg"].copy(); have = np.isfinite(gh)
    a, b, valid = 1.0, 0.0, 0
    if have.sum() >= 8:
        gg[have] = gh[have]                                       # prefix範囲=self実測
        tw_ov = base["gg"][have]; self_ov = gh[have]
        A = np.c_[tw_ov, np.ones(len(tw_ov))]
        try:
            coef, *_ = np.linalg.lstsq(A, self_ov, rcond=None); a, b = float(coef[0]), float(coef[1])
        except Exception:
            a, b = 1.0, 0.0
        if 0.2 <= a <= 5.0 and np.isfinite(a) and np.isfinite(b):
            gg[~have] = a * base["gg"][~have] + b                 # 範囲外=self較正したtwで外挿
            valid = 1
    base = dict(base); base["gg"] = gg
    base["_graft"] = dict(a=float(a), b=float(b), valid=int(valid),
                          cover=float(have.mean()), extrap_frac=float((~have).mean()), nprefix=int(len(kpos)))
    return base


def z_gradient_eval(hw, r=ZGRAD_R):
    """Z傾斜 dZ/dMD を移動平均(断層影響回避)し eval 行のみ返す。"""
    ev = hw["TVT_input"].isna().to_numpy()
    md = hw["MD"].to_numpy(float); z = hw["Z"].to_numpy(float)
    with np.errstate(all="ignore"):
        g = np.gradient(z, md)
    g = pd.Series(g).rolling(2 * int(r) + 1, center=True, min_periods=1).mean().to_numpy()
    return g[ev].astype(np.float32)


# ==================== GRフリー錨 添付 ====================
def attach_anchor(x, wid, physics):
    """PF入力に GRフリー錨(anc/ancs)を添付。physics=False / 錨欠損 / 行数不一致 は錨OFF(ancs=1e9)。
       ★過去struct は使わない。錨源は _ANCHOR(v93 GRフリー, sig=9ft較正)のみ。"""
    n = len(x["md"]); sp = _ANCHOR.get(wid)
    x["_wid"] = wid
    if physics and sp is not None and len(sp["tvt"]) == n:
        x["anc"] = np.asarray(sp["tvt"], float); x["ancs"] = np.asarray(sp["sig"], float)
    else:
        x["anc"] = np.zeros(n); x["ancs"] = np.full(n, 1e9)
    return x


# ==================== GPU 固定ラグ 平滑PF 本体 ====================
def _smoother_core(md, z, gr, valid, grid, glen, vmin, step, gs, ls, ir,
                   anc, ancs, amul, stq, sim, wmap, w_nn, P, N, device, gen, dtype=DTYPE):
    """v52 _v26_smoother_core と数値一致。★v93変更: 錨強度は amul(=P['anchor_mult'])。"""
    B, T = md.shape
    ALPHA = P["mom"]; RN = P["vn"]; PN = P["pn"]; IR = P["init_rate_std"]; IS = P["init_pos_std"]
    RP = P["rp"]; RR = P["rr"]; RESAMP = P["resamp"]; RATE_CLIP = P["rate_clip"]
    GR_POWER = P["gr_power"]; JUMP_PROB = P["jump_prob"]; JUMP_STD = P["jump_std"]; CLIP = P["tvt_clip_margin"]
    PHYS_SIG = P["phys_sig"]; USE_P = P["use_phys"]
    L = int(P.get("smooth_lag", 32)); L = max(0, min(L, T))
    NU = float(P.get("robust_nu", 0.0))                # ★v98: Student-t頑健尤度(0=Gaussian)。v97同様のPF強化をv95側へ移植
    BETA = float(P.get("temper_beta", 1.0))            # ★v98: tempering(1=無効, <1=軟化)

    def rn():
        return torch.randn((B, N), generator=gen, device=device, dtype=dtype)

    pos = ls[:, None] + IS * rn(); rate = ir[:, None] + IR * rn()
    w = torch.full((B, N), 1.0 / N, device=device, dtype=dtype)
    log_lik = torch.zeros(B, device=device, dtype=torch.float64)
    pts_f = torch.zeros((B, T), device=device, dtype=dtype)
    pts_s = torch.zeros((B, T), device=device, dtype=dtype)
    tvt_lo = vmin - CLIP; tvt_hi = vmin + (glen.to(dtype) - 1) * step + CLIP
    glast = torch.gather(grid, 1, (glen - 1).clamp_min(0).unsqueeze(1)); g0 = grid[:, 0:1]
    pm = md[:, 0] - 1.0; arN = torch.arange(N, device=device, dtype=dtype)
    zero = torch.zeros((), device=device, dtype=dtype); dipp = ir[:, None]
    use_sm = L > 0
    if use_sm:
        buf = torch.full((B, L, N), float("nan"), device=device, dtype=dtype); ptr = 0
    for i in range(T):
        act = valid[:, i]; cur = md[:, i]; dm = (cur - pm).clamp_min(1.0)
        rate_n = ALPHA * rate + RN * rn()
        if RATE_CLIP > 0:
            rate_n = rate_n.clamp(-RATE_CLIP, RATE_CLIP)
        pos_n = pos + rate_n * dm[:, None] + PN * rn()
        if JUMP_PROB > 0 and JUMP_STD > 0:
            jm = torch.rand((B, N), generator=gen, device=device, dtype=dtype) < JUMP_PROB
            pos_n = pos_n + torch.where(jm, JUMP_STD * rn(), zero)
        zi = z[:, i][:, None]; tvt = (pos_n - zi).clamp(tvt_lo[:, None], tvt_hi[:, None]); pos_n = tvt + zi
        am = act[:, None]; pos = torch.where(am, pos_n, pos); rate = torch.where(am, rate_n, rate)
        gri = gr[:, i]; obs = act & ~torch.isnan(gri)
        v = pos - zi; ii = (v - vmin[:, None]) / step[:, None]; i0 = torch.floor(ii).long()
        below = i0 < 0; above = i0 >= (glen - 1)[:, None]
        i0c = i0.clamp_min(0); i0c = torch.minimum(i0c, (glen - 2).clamp_min(0)[:, None])
        t = ii - i0c.to(dtype); gA = torch.gather(grid, 1, i0c); gB = torch.gather(grid, 1, i0c + 1)
        eg = gA * (1 - t) + gB * t; eg = torch.where(below, g0, eg); eg = torch.where(above, glast, eg)
        d = ((gri[:, None] - eg) / gs[:, None]).abs(); dp = d ** GR_POWER
        if NU > 0.0:                                   # ★v98: Student-t 頑健尤度(重い裾)
            lk = (1.0 + dp / NU) ** (-(NU + 1.0) * 0.5)
        else:
            lk = torch.where(dp < 600.0, torch.exp(-0.5 * dp), zero)
        if BETA != 1.0:                                # ★v98: tempering(尤度軟化)
            lk = lk ** BETA
        lk = lk.clamp_min(1e-300)
        avg = (w * lk).sum(1)
        log_lik = log_lik + torch.where(obs, torch.log(avg.clamp_min(1e-300).double()),
                                        torch.zeros((), device=device, dtype=torch.float64))
        if USE_P:
            dphys = (rate - dipp) / PHYS_SIG; lkp = torch.exp(-0.5 * dphys * dphys).clamp_min(1e-300)
        else:
            lkp = torch.ones_like(w)
        # ★v93: GRフリー行錨(mult = amul = P['anchor_mult'])。非physicsは ancs=1e9 で無効化。
        asig = (ancs[:, i][:, None] * amul).clamp_min(1e-6)
        da = (v - anc[:, i][:, None]) / asig
        lka = torch.where(da * da < 1200.0, torch.exp(-0.5 * da * da), zero).clamp_min(1e-300)
        # NN-emission(学習類似度を温度 w_nn で乗算)。帯中心 stq は GRフリー錨tvt(module2)。
        if w_nn > 0:
            jj = torch.round((v - stq[:, i][:, None]) / SIM_STEP_NN).long() + NBAND_NN
            inb = (jj >= 0) & (jj <= 2 * NBAND_NN)
            sim_i = sim[:, i, :].to(dtype)[wmap]
            sv = torch.gather(sim_i, 1, jj.clamp(0, 2 * NBAND_NN))
            sv = torch.where(inb, sv, torch.full_like(sv, -1.0))
            lka = lka * torch.exp(w_nn * sv)
        w_new = torch.where(obs[:, None], w * lk * lkp * lka, w * lkp * lka)
        ws = w_new.sum(1, keepdim=True)
        w = torch.where(ws > 0, w_new / ws.clamp_min(1e-300), torch.full_like(w, 1.0 / N))
        ne = (w * w).sum(1); need = act & ((1.0 / ne) < (RESAMP * N))
        if bool(need.any()):
            cumw = torch.cumsum(w, 1); u0 = torch.rand((B, 1), generator=gen, device=device, dtype=dtype) * (1.0 / N)
            u = u0 + arN[None, :] / N; idx = torch.searchsorted(cumw, u, right=False).clamp(0, N - 1)
            pos_rs = torch.gather(pos, 1, idx) + RP * rn(); rate_rs = torch.gather(rate, 1, idx) + RR * rn()
            if RATE_CLIP > 0:
                rate_rs = rate_rs.clamp(-RATE_CLIP, RATE_CLIP)
            nm = need[:, None]; pos = torch.where(nm, pos_rs, pos); rate = torch.where(nm, rate_rs, rate)
            w = torch.where(nm, torch.full_like(w, 1.0 / N), w)
            if use_sm:
                buf_g = torch.gather(buf, 2, idx[:, None, :].expand(B, L, N))
                buf = torch.where(need[:, None, None], buf_g, buf)
        pts_f[:, i] = (w * (pos - zi)).sum(1)
        if use_sm:
            if i >= L:
                old = buf[:, ptr, :]; pts_s[:, i - L] = (w * old).sum(1) - z[:, i - L]
            buf[:, ptr, :] = pos; ptr = (ptr + 1) % L
        pm = torch.where(act, cur, pm)
    if use_sm:
        for j in range(max(0, T - L), T):
            pts_s[:, j] = (w * buf[:, j % L, :]).sum(1) - z[:, j]
    else:
        pts_s = pts_f
    return pts_f, pts_s, log_lik


def _smoother_core_full(md, z, gr, valid, grid, glen, vmin, step, gs, ls, ir,
                        anc, ancs, amul, stq, sim, wmap, w_nn, P, N, device, gen, dtype=DTYPE):
    """★v99 full平滑: forward は _smoother_core と数値一致。固定ラグbufの代わりに全履歴(pos fp32=錨delta / anc int16)を
       保存し、単一backward祖先sweep(最終重み)で全区間平滑 pts_s を返す。robust/temper/phys/anchor/NN-emission 全対応。"""
    B, T = md.shape
    ALPHA = P["mom"]; RN = P["vn"]; PN = P["pn"]; IR = P["init_rate_std"]; IS = P["init_pos_std"]
    RP = P["rp"]; RR = P["rr"]; RESAMP = P["resamp"]; RATE_CLIP = P["rate_clip"]
    GR_POWER = P["gr_power"]; JUMP_PROB = P["jump_prob"]; JUMP_STD = P["jump_std"]; CLIP = P["tvt_clip_margin"]
    PHYS_SIG = P["phys_sig"]; USE_P = P["use_phys"]
    NU = float(P.get("robust_nu", 0.0)); BETA = float(P.get("temper_beta", 1.0))

    def rn():
        return torch.randn((B, N), generator=gen, device=device, dtype=dtype)
    pos = ls[:, None] + IS * rn(); rate = ir[:, None] + IR * rn()
    w = torch.full((B, N), 1.0 / N, device=device, dtype=dtype)
    log_lik = torch.zeros(B, device=device, dtype=torch.float64)
    pts_f = torch.zeros((B, T), device=device, dtype=dtype)
    pos_hist = torch.empty((T, B, N), device=device, dtype=dtype)          # fp32(錨lsからのdelta)
    anc_hist = torch.empty((T, B, N), device=device, dtype=torch.int16)     # 祖先index
    lsr = ls[:, None]
    tvt_lo = vmin - CLIP; tvt_hi = vmin + (glen.to(dtype) - 1) * step + CLIP
    glast = torch.gather(grid, 1, (glen - 1).clamp_min(0).unsqueeze(1)); g0 = grid[:, 0:1]
    pm = md[:, 0] - 1.0; arN = torch.arange(N, device=device, dtype=dtype)
    arL = torch.arange(N, device=device, dtype=torch.long)[None, :].expand(B, N)
    zero = torch.zeros((), device=device, dtype=dtype); dipp = ir[:, None]
    for i in range(T):
        act = valid[:, i]; cur = md[:, i]; dm = (cur - pm).clamp_min(1.0)
        rate_n = ALPHA * rate + RN * rn()
        if RATE_CLIP > 0:
            rate_n = rate_n.clamp(-RATE_CLIP, RATE_CLIP)
        pos_n = pos + rate_n * dm[:, None] + PN * rn()
        if JUMP_PROB > 0 and JUMP_STD > 0:
            jm = torch.rand((B, N), generator=gen, device=device, dtype=dtype) < JUMP_PROB
            pos_n = pos_n + torch.where(jm, JUMP_STD * rn(), zero)
        zi = z[:, i][:, None]; tvt = (pos_n - zi).clamp(tvt_lo[:, None], tvt_hi[:, None]); pos_n = tvt + zi
        am = act[:, None]; pos = torch.where(am, pos_n, pos); rate = torch.where(am, rate_n, rate)
        gri = gr[:, i]; obs = act & ~torch.isnan(gri)
        v = pos - zi; ii = (v - vmin[:, None]) / step[:, None]; i0 = torch.floor(ii).long()
        below = i0 < 0; above = i0 >= (glen - 1)[:, None]
        i0c = i0.clamp_min(0); i0c = torch.minimum(i0c, (glen - 2).clamp_min(0)[:, None])
        t = ii - i0c.to(dtype); gA = torch.gather(grid, 1, i0c); gB = torch.gather(grid, 1, i0c + 1)
        eg = gA * (1 - t) + gB * t; eg = torch.where(below, g0, eg); eg = torch.where(above, glast, eg)
        d = ((gri[:, None] - eg) / gs[:, None]).abs(); dp = d ** GR_POWER
        if NU > 0.0:
            lk = (1.0 + dp / NU) ** (-(NU + 1.0) * 0.5)
        else:
            lk = torch.where(dp < 600.0, torch.exp(-0.5 * dp), zero)
        if BETA != 1.0:
            lk = lk ** BETA
        lk = lk.clamp_min(1e-300)
        avg = (w * lk).sum(1)
        log_lik = log_lik + torch.where(obs, torch.log(avg.clamp_min(1e-300).double()),
                                        torch.zeros((), device=device, dtype=torch.float64))
        if USE_P:
            dphys = (rate - dipp) / PHYS_SIG; lkp = torch.exp(-0.5 * dphys * dphys).clamp_min(1e-300)
        else:
            lkp = torch.ones_like(w)
        asig = (ancs[:, i][:, None] * amul).clamp_min(1e-6)
        da = (v - anc[:, i][:, None]) / asig
        lka = torch.where(da * da < 1200.0, torch.exp(-0.5 * da * da), zero).clamp_min(1e-300)
        if w_nn > 0:
            jj = torch.round((v - stq[:, i][:, None]) / SIM_STEP_NN).long() + NBAND_NN
            inb = (jj >= 0) & (jj <= 2 * NBAND_NN)
            sim_i = sim[:, i, :].to(dtype)[wmap]
            sv = torch.gather(sim_i, 1, jj.clamp(0, 2 * NBAND_NN))
            sv = torch.where(inb, sv, torch.full_like(sv, -1.0))
            lka = lka * torch.exp(w_nn * sv)
        w_new = torch.where(obs[:, None], w * lk * lkp * lka, w * lkp * lka)
        ws = w_new.sum(1, keepdim=True)
        w = torch.where(ws > 0, w_new / ws.clamp_min(1e-300), torch.full_like(w, 1.0 / N))
        ne = (w * w).sum(1); need = act & ((1.0 / ne) < (RESAMP * N))
        anc_step = arL
        if bool(need.any()):
            cumw = torch.cumsum(w, 1); u0 = torch.rand((B, 1), generator=gen, device=device, dtype=dtype) * (1.0 / N)
            u = u0 + arN[None, :] / N; idx = torch.searchsorted(cumw, u, right=False).clamp(0, N - 1)
            pos_rs = torch.gather(pos, 1, idx) + RP * rn(); rate_rs = torch.gather(rate, 1, idx) + RR * rn()
            if RATE_CLIP > 0:
                rate_rs = rate_rs.clamp(-RATE_CLIP, RATE_CLIP)
            nm = need[:, None]; pos = torch.where(nm, pos_rs, pos); rate = torch.where(nm, rate_rs, rate)
            w = torch.where(nm, torch.full_like(w, 1.0 / N), w)
            anc_step = torch.where(nm, idx, arL)
        pts_f[:, i] = (w * (pos - zi)).sum(1)
        pos_hist[i] = pos - lsr; anc_hist[i] = anc_step.to(torch.int16)
        pm = torch.where(act, cur, pm)
    pts_s = torch.zeros((B, T), device=device, dtype=dtype); a = arL.clone(); wfin = w
    for i in range(T - 1, -1, -1):
        pts_s[:, i] = (wfin * (torch.gather(pos_hist[i], 1, a).to(dtype) + lsr)).sum(1) - z[:, i]
        a = torch.gather(anc_hist[i], 1, a).long()
    del pos_hist, anc_hist
    return pts_f, pts_s, log_lik


def _pad(inps, device, dtype=DTYPE):
    """可変長wellをバッチテンソルへ。anc/ancs/stq/sim(NN)も詰める。sim帯は入力dictの _sim/_st から。"""
    W = len(inps); Tmax = max(len(x["md"]) for x in inps); Gmax = max(len(x["gg"]) for x in inps)
    md = torch.zeros((W, Tmax), dtype=dtype); z = torch.zeros((W, Tmax), dtype=dtype)
    gr = torch.full((W, Tmax), float("nan"), dtype=dtype); valid = torch.zeros((W, Tmax), dtype=torch.bool)
    grid = torch.zeros((W, Gmax), dtype=dtype); glen = torch.zeros(W, dtype=torch.long)
    vmin = torch.zeros(W, dtype=dtype); step = torch.zeros(W, dtype=dtype); gs = torch.zeros(W, dtype=dtype)
    ls = torch.zeros(W, dtype=dtype); ir = torch.zeros(W, dtype=dtype)
    anc = torch.zeros((W, Tmax), dtype=dtype); ancs = torch.full((W, Tmax), 1e9, dtype=dtype)
    stq = torch.zeros((W, Tmax), dtype=dtype)
    simt = torch.full((W, Tmax, 2 * NBAND_NN + 1), -1.0, dtype=torch.float16)
    for b, x in enumerate(inps):
        Tn = len(x["md"]); G = len(x["gg"])
        _sim = x.get("_sim"); _st = x.get("_st")
        if _sim is not None and _st is not None:
            _Ts = min(Tn, len(_st))
            stq[b, :_Ts] = torch.from_numpy(np.asarray(_st[:_Ts], np.float32))
            if _Ts < Tmax:
                stq[b, _Ts:] = stq[b, _Ts - 1]
            simt[b, :_Ts] = torch.from_numpy(np.asarray(_sim[:_Ts], np.float16))
        if "anc" in x:
            anc[b, :Tn] = torch.from_numpy(x["anc"].astype("float32"))
            if Tn < Tmax:
                anc[b, Tn:] = anc[b, Tn - 1]
            ancs[b, :Tn] = torch.from_numpy(x["ancs"].astype("float32"))
        md[b, :Tn] = torch.from_numpy(x["md"].astype("float32"))
        if Tn < Tmax:
            md[b, Tn:] = md[b, Tn - 1]
        z[b, :Tn] = torch.from_numpy(x["z"].astype("float32"))
        gr[b, :Tn] = torch.from_numpy(x["gr"].astype("float32")); valid[b, :Tn] = True
        grid[b, :G] = torch.from_numpy(x["gg"].astype("float32"))
        if G < Gmax:
            grid[b, G:] = grid[b, G - 1]
        glen[b] = G; vmin[b] = x["gmin"]; step[b] = x["gst"]; gs[b] = x["gs"]; ls[b] = x["ls"]; ir[b] = x["ir"]
    to = lambda tt: tt.to(device)
    return dict(md=to(md), z=to(z), gr=to(gr), valid=to(valid), grid=to(grid), glen=to(glen),
                vmin=to(vmin), step=to(step), gs=to(gs), ls=to(ls), ir=to(ir), anc=to(anc), ancs=to(ancs),
                stq=to(stq), simt=to(simt))


def _pf_devices(chunk_env_default):
    """PF実行デバイス一覧を決定。
       PF_SIM_NGPU=N(>=2): 1GPU上でN論理分割=分割/結合/スレッド機構の検証用(全て同一物理GPU)。
       PF_NGPU=N(>=2): 実GPUをN枚使用(Kaggle 2×T4)。既定=[DEVICE]で現行完全不変。"""
    sim_ng = int(os.environ.get("PF_SIM_NGPU", "0"))
    if sim_ng >= 2:
        return [DEVICE] * sim_ng, True
    ng = int(os.environ.get("PF_NGPU", "1"))
    if ng >= 2 and torch.cuda.is_available() and torch.cuda.device_count() >= 2:
        return ["cuda:%d" % i for i in range(min(ng, torch.cuda.device_count()))], False
    return [DEVICE], False


def run_smoother_ext(inps, P, seed, n_seeds, chunk, w_nn=0.0, capture=False):
    """各 well: smoothed(seed尤度加重平均)と std(seed間加重std)を返す。
       ★v93: 錨強度 amul は P['anchor_mult'](pfA=20)。非physicsバンクは ancs=1e9 で錨無効。
       ★capture=True: v34モードゲート用に per-seed 平滑ps(S,T)/forward pfw/ll/ww も out に付与(RAM増)。
       ★マルチGPU: whole-chunkを各デバイスへラウンドロビン分配しスレッド並列。seedはchunk毎リセット・
         torch CUDA RNGはデバイス非依存なので、同一chunk境界なら単一GPUと(浮動小数の非決定性を除き)一致。"""
    import threading
    N = int(P["n_particles"]); S = n_seeds; W = len(inps); out = [None] * W
    ls_scale = float(P["likelihood_scale"]); amul = float(P.get("anchor_mult", 1.0))
    ch_env = os.environ.get("PF_WELL_CHUNK")
    if ch_env:
        chunk = int(ch_env)

    # ★v99 full平滑モード: 固定ラグの代わりに全区間系譜平滑。可変chunk(VRAM予算)・長さソート。
    SMOOTH_MODE = str(P.get("smooth_mode", os.environ.get("SMOOTH_MODE", "fixedlag")))
    if SMOOTH_MODE == "full":
        budget = float(os.environ.get("FULL_VRAM_GB", "8.0")) * 1e9
        order = sorted(range(W), key=lambda i: len(inps[i]["md"]))
        chunks = []; p = 0                                       # ★可変chunkを全部先に構築(デバイス分配用)
        while p < W:
            Tmax = len(inps[order[p]]["md"]); nw = 0
            while p + nw < W and nw < 64:
                tmx = max(Tmax, len(inps[order[p + nw]]["md"]))
                if (nw + 1) * S * tmx * N * 6 > budget and nw >= 1:
                    break
                Tmax = tmx; nw += 1
            chunks.append(order[p:p + nw]); p += nw

        def _proc_full(sel, device):                            # ★1可変chunkを指定deviceで処理(seedはchunk毎リセット=device非依存)
            sub = [inps[j] for j in sel]; nw = len(sel); pad = _pad(sub, device)
            rep = lambda tt: tt.repeat_interleave(S, 0)
            gen = torch.Generator(device=device); gen.manual_seed(seed)
            _wmap = torch.arange(nw, device=device).repeat_interleave(S)
            pf, ps, ll = _smoother_core_full(
                rep(pad["md"]), rep(pad["z"]), rep(pad["gr"]), rep(pad["valid"]), rep(pad["grid"]),
                pad["glen"].repeat_interleave(S), pad["vmin"].repeat_interleave(S), pad["step"].repeat_interleave(S),
                pad["gs"].repeat_interleave(S), pad["ls"].repeat_interleave(S), pad["ir"].repeat_interleave(S),
                rep(pad["anc"]), rep(pad["ancs"]), amul, rep(pad["stq"]), pad["simt"], _wmap, w_nn, P, N, device, gen)
            ps = ps.view(nw, S, -1).double().cpu().numpy(); pf = pf.view(nw, S, -1).double().cpu().numpy(); ll = ll.view(nw, S).cpu().numpy()
            for j in range(nw):
                gi = sel[j]; T = len(sub[j]["md"]); llw = ll[j]
                if np.isfinite(llw).any():
                    mx = np.nanmax(llw[np.isfinite(llw)]); lk = np.where(np.isfinite(llw), llw - mx, -np.inf)
                    wwv = np.exp(lk / max(ls_scale, 1e-6)); s = wwv.sum(); wwv = wwv / s if s > 1e-300 else np.full(S, 1.0 / S)
                else:
                    wwv = np.full(S, 1.0 / S)
                wwv = _ps_combo_reweight(wwv, ps[j, :, :T], sub[j].get("_st"), float(P.get("_ps_combo_tau", 0.0)))
                smean = (wwv[:, None] * ps[j, :, :T]).sum(0)
                sstd = np.sqrt(np.maximum((wwv[:, None] * (ps[j, :, :T] - smean[None, :]) ** 2).sum(0), 0.0))
                nobs = int(np.isfinite(sub[j]["gr"][:T]).sum())
                llbest = float(np.nanmax(llw[np.isfinite(llw)])) if np.isfinite(llw).any() else np.nan
                out[gi] = dict(mean=smean.astype(np.float32), std=sstd.astype(np.float32), loglik=np.float32(llbest / max(nobs, 1)))
                if capture:
                    out[gi].update(ps=ps[j, :, :T].astype(np.float32), pfw=pf[j, :, :T].astype(np.float32),
                                   ll=llw.astype(np.float64), ww=wwv.astype(np.float64))

        fdevices, fsim = _pf_devices(chunk)                     # ★PF_NGPU>=2で実GPU複数(既定=[DEVICE]で単GPU=現行不変)
        if len(fdevices) <= 1:
            for sel in chunks:
                _proc_full(sel, fdevices[0])
        else:
            assign = {i: [] for i in range(len(fdevices))}      # 可変chunkをデバイスへラウンドロビン
            for i, sel in enumerate(chunks):
                assign[i % len(fdevices)].append(sel)
            errs = []
            def _wf(di):
                try:
                    for sel in assign[di]:
                        _proc_full(sel, fdevices[di])
                except Exception as e:
                    errs.append(e)
            ths = [threading.Thread(target=_wf, args=(di,)) for di in range(len(fdevices))]
            for t in ths: t.start()
            for t in ths: t.join()
            if errs:
                raise errs[0]
            print(f"[fullPF-multiGPU] devices={fdevices} sim={fsim} chunks={len(chunks)} wells={W}", flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return out

    devices, sim = _pf_devices(chunk)

    def _proc(c0, device):
        sub = inps[c0:c0 + chunk]; w = len(sub); pad = _pad(sub, device)
        rep = lambda tt: tt.repeat_interleave(S, 0)
        gen = torch.Generator(device=device); gen.manual_seed(seed)
        _wmap = torch.arange(w, device=device).repeat_interleave(S)
        pf, ps, ll = _smoother_core(
            rep(pad["md"]), rep(pad["z"]), rep(pad["gr"]), rep(pad["valid"]), rep(pad["grid"]),
            pad["glen"].repeat_interleave(S), pad["vmin"].repeat_interleave(S), pad["step"].repeat_interleave(S),
            pad["gs"].repeat_interleave(S), pad["ls"].repeat_interleave(S), pad["ir"].repeat_interleave(S),
            rep(pad["anc"]), rep(pad["ancs"]), amul, rep(pad["stq"]), pad["simt"], _wmap, w_nn, P, N, device, gen)
        pf = pf.view(w, S, -1).double().cpu().numpy(); ps = ps.view(w, S, -1).double().cpu().numpy()
        ll = ll.view(w, S).cpu().numpy()
        for j in range(w):
            T = len(sub[j]["md"]); llw = ll[j]
            if np.isfinite(llw).any():
                mx = np.nanmax(llw[np.isfinite(llw)]); lk = np.where(np.isfinite(llw), llw - mx, -np.inf)
                ww = np.exp(lk / max(ls_scale, 1e-6)); s = ww.sum()
                ww = ww / s if s > 1e-300 else np.full(S, 1.0 / S)
            else:
                ww = np.full(S, 1.0 / S)
            ww = _ps_combo_reweight(ww, ps[j, :, :T], sub[j].get("_st"), float(P.get("_ps_combo_tau", 0.0)))
            smean = (ww[:, None] * ps[j, :, :T]).sum(0)
            sstd = np.sqrt(np.maximum((ww[:, None] * (ps[j, :, :T] - smean[None, :]) ** 2).sum(0), 0.0))
            nobs = int(np.isfinite(sub[j]["gr"][:T]).sum())      # ★GR観測行数(loglik正規化)
            llbest = float(np.nanmax(llw[np.isfinite(llw)])) if np.isfinite(llw).any() else np.nan
            out[c0 + j] = dict(mean=smean.astype(np.float32), std=sstd.astype(np.float32),
                               loglik=np.float32(llbest / max(nobs, 1)))   # per-row平均対数尤度=その表現PFのGR適合
            if capture:                                                    # ★v34ゲート: per-seed 生軌跡
                out[c0 + j].update(ps=ps[j, :, :T].astype(np.float32), pfw=pf[j, :, :T].astype(np.float32),
                                   ll=llw.astype(np.float64), ww=ww.astype(np.float64))

    starts = list(range(0, W, chunk))
    if len(devices) <= 1:
        for c0 in starts:                                        # ★単一GPU経路=現行と完全同一(検証可能)
            _proc(c0, devices[0])
    else:
        assign = {i: [] for i in range(len(devices))}            # chunkをデバイスへラウンドロビン割当
        for i, c0 in enumerate(starts):
            assign[i % len(devices)].append(c0)
        errs = []
        def _worker(di):
            try:
                for c0 in assign[di]:
                    _proc(c0, devices[di])
            except Exception as e:
                errs.append(e)
        ths = [threading.Thread(target=_worker, args=(di,)) for di in range(len(devices))]
        for t in ths: t.start()
        for t in ths: t.join()
        if errs:
            raise errs[0]
        print(f"[PF-multiGPU] devices={devices} sim={sim} chunks={len(starts)} wells={W}", flush=True)
    if torch.cuda.is_available():
        for d in set(devices):
            if str(d).startswith("cuda"):
                torch.cuda.synchronize(d)
    return out


# ==================== 単体検証(python pf_banks_v93.py) ====================

# %% [markdown]
# ## 8. Run pfA × twGR only
# Exactly one bank and one representation are allowed. Each well uses 600 particles × 32 seeds, full ancestral smoothing, physical anchor mult 20, and learned-emission weight 0.01.

# %%
public_config = json.loads(config_path.read_text(encoding="utf-8"))
if public_config["bank_order"] != ["pf_1", "pf_2", "pf_3", "r0_seed32", "r1_seed32", "pfA"]:
    raise RuntimeError("unexpected public v96 bank order")
P = bank_param(PF_BANK)
expected_contract = {
    "n_particles": PF_N_PARTICLES,
    "smooth_mode": PF_SMOOTH_MODE,
    "anchor_mult": 20.0,
    "grid_step": 0.2,
}
for key, expected in expected_contract.items():
    actual = P.get(key)
    if actual != expected:
        raise RuntimeError(f"pfA contract drift for {key}: {actual!r} != {expected!r}")
if N_SEED != PF_N_SEEDS or P["_w_nn"] != 0.01 or not P["_physics"]:
    raise RuntimeError("pfA seed/emission/anchor contract drift")

sample_keys = sample["id"].astype(str)
parsed = sample_keys.str.rsplit("_", n=1, expand=True)
if parsed.shape[1] != 2:
    raise RuntimeError("sample ids must end in an underscore-separated row index")
sample_wells = parsed[0].astype(str)
try:
    sample_rows = parsed[1].astype(int)
except Exception as exc:
    raise RuntimeError("sample id row suffix is not integer") from exc

test_dir = DATA_ROOT / "test"
available_wells = sorted({path.name.split("__", 1)[0] for path in test_dir.glob("*__horizontal_well.csv")})
requested_wells = sorted(sample_wells.unique())
if set(requested_wells) - set(available_wells):
    raise RuntimeError(f"sample wells missing from runtime test files: {sorted(set(requested_wells)-set(available_wells))}")

names: list[str] = []
inps: list[dict] = []
eval_rows_by_well: dict[str, np.ndarray] = {}
for wid in requested_wells:
    hw = pd.read_csv(test_dir / f"{wid}__horizontal_well.csv")
    tw = pd.read_csv(test_dir / f"{wid}__typewell.csv").sort_values("TVT")
    for column in ["MD", "Z", "GR", "TVT_input"]:
        if column not in hw.columns:
            raise RuntimeError(f"{wid} horizontal well missing {column}")
    for column in ["TVT", "GR"]:
        if column not in tw.columns:
            raise RuntimeError(f"{wid} typewell missing {column}")
    x = build_smoother_inputs(hw, tw["TVT"].to_numpy(float), tw["GR"].to_numpy(float), P)
    if x is None:
        raise RuntimeError(f"{wid} has no hidden suffix for PF")
    x = attach_anchor(x, wid, P["_physics"])
    similarity = SIMD.get(wid)
    if similarity is None or len(similarity.get("st", [])) != len(x["md"]):
        raise RuntimeError(f"{wid} learned-emission similarity missing or row-mismatched")
    x["_sim"] = similarity["sim"]
    x["_st"] = similarity["st"]
    names.append(wid)
    inps.append(x)
    eval_rows_by_well[wid] = np.flatnonzero(hw["TVT_input"].isna().to_numpy())

pf_started = time.perf_counter()
outs = run_smoother_ext(
    inps,
    P,
    seed=PF_GENERATION_SEED,
    n_seeds=PF_N_SEEDS,
    chunk=PF_WELL_CHUNK,
    w_nn=P["_w_nn"],
)
pf_seconds = time.perf_counter() - pf_started
if len(outs) != len(names) or any(output is None for output in outs):
    raise RuntimeError("PF did not return exactly one output for every requested well")

prediction_lookup: dict[str, tuple[float, float, float]] = {}
for wid, rows, output in zip(names, (eval_rows_by_well[w] for w in names), outs):
    mean = np.asarray(output["mean"], dtype=np.float64)
    std = np.asarray(output["std"], dtype=np.float64)
    if len(mean) != len(rows) or len(std) != len(rows):
        raise RuntimeError(f"{wid} PF output length mismatch")
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise RuntimeError(f"{wid} PF output contains non-finite values")
    loglik = float(output["loglik"])
    for row_idx, mean_value, std_value in zip(rows, mean, std):
        prediction_lookup[f"{wid}_{int(row_idx)}"] = (float(mean_value), float(std_value), loglik)

expected_ids = set(sample_keys)
actual_ids = set(prediction_lookup)
if expected_ids != actual_ids:
    raise RuntimeError(
        f"PF/sample id mismatch: missing={len(expected_ids-actual_ids)} extra={len(actual_ids-expected_ids)}"
    )
print({
    "bank": PF_BANK,
    "representation": PF_REPRESENTATION,
    "wells": len(names),
    "rows": len(prediction_lookup),
    "particles": PF_N_PARTICLES,
    "seeds": PF_N_SEEDS,
    "smooth_mode": PF_SMOOTH_MODE,
    "pf_seconds": pf_seconds,
})

# %% [markdown]
# ## 9. Sample-ID alignment, manifests, and LATE SUBMIT output
# The runtime sample submission is the sole schema/order authority. The direct pfA smoothed mean is written without fusion, gain, or visible-test branch. Logical candidate content and final submission receive SHA256 manifests.

# %%
candidate = pd.DataFrame({
    "id": sample_keys,
    "pfa_tw_mean": [prediction_lookup[key][0] for key in sample_keys],
    "pfa_tw_std": [prediction_lookup[key][1] for key in sample_keys],
    "pfa_tw_loglik": [prediction_lookup[key][2] for key in sample_keys],
})
submission = sample[["id"]].copy()
submission[TARGET_COLUMN] = candidate["pfa_tw_mean"].to_numpy(dtype=np.float64)

if len(submission) != len(sample) or not submission["id"].equals(sample["id"]):
    raise RuntimeError("submission id/order/row contract failed")
if submission["id"].astype(str).duplicated().any():
    raise RuntimeError("submission contains duplicate ids")
if not np.isfinite(submission[TARGET_COLUMN].to_numpy(dtype=np.float64)).all():
    raise RuntimeError("submission contains non-finite predictions")

candidate_content = candidate.to_csv(index=False, lineterminator="\n")
candidate_gzip_path = WORK_ROOT / "pfa_tw_candidate.csv.gz"
with gzip.open(candidate_gzip_path, "wt", encoding="utf-8", newline="") as handle:
    handle.write(candidate_content)
submission_path = WORK_ROOT / "submission.csv"
submission.to_csv(submission_path, index=False, lineterminator="\n")

checkpoint_manifest = [
    {
        "name": name,
        "source_path": str(checkpoints[name]),
        "sha256": sha256_path(NN50_ROOT / name),
        "bytes": int((NN50_ROOT / name).stat().st_size),
    }
    for name in sorted(checkpoints)
]
model_manifest = {
    "experiment": EXPERIMENT,
    "late_submission_phase": LATE_SUBMISSION_PHASE,
    "public_encoder_models": checkpoint_manifest,
    "anchor_fold_artifact": {
        "path": str(fold_artifact_path),
        "sha256": sha256_path(fold_artifact_path),
        "bytes": int(fold_artifact_path.stat().st_size),
        "folds": 5,
        "seeds_per_fold": 3,
        "model_count": 15,
        "loss": "masked_huber_delta_8ft",
    },
    "pf": {
        "bank": PF_BANK,
        "representation": PF_REPRESENTATION,
        "config_sha256": sha256_path(config_path),
        "particles": PF_N_PARTICLES,
        "seeds": PF_N_SEEDS,
        "smooth_mode": PF_SMOOTH_MODE,
        "generation_seed": PF_GENERATION_SEED,
    },
}
(WORK_ROOT / "model_manifest.json").write_text(json.dumps(model_manifest, indent=2, sort_keys=True), encoding="utf-8")

manifest = {
    "experiment": EXPERIMENT,
    "late_submission": True,
    "late_submission_phase": LATE_SUBMISSION_PHASE,
    "submission_message": "LATE SUBMIT | exp516 | 6th-place pfA x twGR standalone faithful replay | fixed v1",
    "source": {
        "public_kernel": PUBLIC_KERNEL,
        "public_kernel_id_no": PUBLIC_KERNEL_ID_NO,
        "public_notebook_sha256": PUBLIC_NOTEBOOK_SHA256,
        "public_anchor_source_sha256": PUBLIC_ANCHOR_SOURCE_SHA256,
        "public_emission_source_sha256": PUBLIC_EMISSION_SOURCE_SHA256,
        "adapted_emission_source_sha256": ADAPTED_EMISSION_SOURCE_SHA256,
        "public_pf_source_sha256": PUBLIC_PF_SOURCE_SHA256,
        "public_config_sha256": PUBLIC_CONFIG_SHA256,
        "public_config_embedded_text_sha256": PUBLIC_CONFIG_TEXT_SHA256,
    },
    "runtime": {
        "gpu_names": gpu_names,
        "anchor_seconds": anchor_seconds,
        "similarity_seconds": similarity_seconds,
        "pf_seconds": pf_seconds,
        "well_count": len(names),
        "row_count": len(candidate),
    },
    "artifacts": {
        "anchor_sha256": sha256_path(anchor_path),
        "anchor_fold_artifact_sha256": sha256_path(fold_artifact_path),
        "similarity_sha256": sha256_path(similarity_path),
        "candidate_decompressed_content_sha256": sha256_text(candidate_content),
        "candidate_raw_gzip_sha256": sha256_path(candidate_gzip_path),
        "prediction_content_sha256": logical_csv_sha(candidate[["id", "pfa_tw_mean"]]),
        "submission_sha256": sha256_path(submission_path),
    },
    "submission_contract": {
        "columns": list(submission.columns),
        "rows": int(len(submission)),
        "duplicate_ids": int(submission["id"].astype(str).duplicated().sum()),
        "missing_predictions": int(submission[TARGET_COLUMN].isna().sum()),
        "finite": bool(np.isfinite(submission[TARGET_COLUMN].to_numpy(dtype=np.float64)).all()),
        "sample_id_order_exact": bool(submission["id"].equals(sample["id"])),
    },
    "prediction_summary": {
        "mean": float(submission[TARGET_COLUMN].mean()),
        "std": float(submission[TARGET_COLUMN].std()),
        "min": float(submission[TARGET_COLUMN].min()),
        "max": float(submission[TARGET_COLUMN].max()),
    },
    "deterministic_anchor": False,
    "deterministic_reason": "GPU fixed-environment rerun equality not yet tested",
}
(WORK_ROOT / "exp516_execution_manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
)
print(json.dumps({
    "LATE_SUBMIT": True,
    "submission": str(submission_path),
    "submission_sha256": manifest["artifacts"]["submission_sha256"],
    "candidate_content_sha256": manifest["artifacts"]["candidate_decompressed_content_sha256"],
    "contract": manifest["submission_contract"],
}, indent=2))

