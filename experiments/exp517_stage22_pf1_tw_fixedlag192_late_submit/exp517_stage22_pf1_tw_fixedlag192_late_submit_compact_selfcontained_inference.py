# %% [markdown]
# # LATE SUBMIT — exp517 stage 2-2 pf_1 × twGR fixed-lag-192 proxy
#
# This is a post-competition component audit, not an official-place submission.
# The published stage 2-2 score used five PF inputs plus a tabular model; this notebook emits one PF path only.

# %% [markdown]
# ## Contents
# 1. Imports and immutable proxy contract
# 2. Runtime and dynamic input guards
# 3. Public GPU particle-filter and fixed-lag smoother engine
# 4. Run pf_1 × twGR × fixed-lag 192 only
# 5. Sample-ID alignment, manifests, and LATE SUBMIT output

# %% [markdown]
# ## 1. Imports and immutable proxy contract
# The public final notebook and v96 config are SHA-pinned. Stage 2-2 exact parameters are not public, so the final released pf_1 parameters are an explicitly approved proxy. No author stage 2-2 score is treated as this run's target.

# %%
from __future__ import annotations

import gzip
import hashlib
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

EXPERIMENT = "exp517_stage22_pf1_tw_fixedlag192_late_submit"
LATE_SUBMISSION_PHASE = "post_competition_late_submission"
FIDELITY = "proxy"
PUBLIC_KERNEL = "k256net/public20th-private6th-pf-pf-pf-pf-and-bagging"
PUBLIC_KERNEL_ID_NO = 126919690
PUBLIC_NOTEBOOK_SHA256 = "b44f7889d6abdf9b027d33cb6c6b45f23902d609fae8d06f332914017784c924"
PUBLIC_CONFIG_SHA256 = "80e973d5f5e0e39be758a03f399cdd3d81d9e79320da8db6fbddbc25c2a202f3"
PUBLIC_CONFIG_TEXT_SHA256 = "aff2bcf63d1dd0b24ceefcd77a8b4fc058e0977e4e9518b58c6fea8c5468d962"
PUBLIC_PF_SOURCE_SHA256 = "ea5e5af2a6fe6e344ad3a792c2735368dbe9f2f61aae79b804abe3eb493e6a6e"
PF_BANK = "pf_1"
PF_REPRESENTATION = "tw"
PF_GENERATION_SEED = 4423098
PF_N_PARTICLES = 600
PF_N_SEEDS = 32
PF_SMOOTH_MODE = "fixedlag"
PF_SMOOTH_LAG = 192
PF_WELL_CHUNK = 40
PF_PHYSICS_ENABLED = False
PF_EMISSION_WEIGHT = 0.0
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
# ## 2. Runtime and dynamic input guards
# The current competition test and sample submission are discovered dynamically. An empty anchor payload is materialized only because the released PF module loads that file at import; pf_1 is non-physical and never consumes it.

# %%
def resolve_competition_root() -> Path:
    roots = [
        Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction"),
        Path("/kaggle/input/rogii-wellbore-geology-prediction"),
    ]
    roots.extend(path.parent for path in Path("/kaggle/input").rglob("sample_submission.csv"))
    roots.append(Path("data/raw").resolve())
    for root in roots:
        if (root / "test").is_dir() and (root / "sample_submission.csv").is_file():
            return root
    raise FileNotFoundError("competition root with test/sample_submission.csv was not found")


DATA_ROOT = resolve_competition_root()
SAMPLE_PATH = DATA_ROOT / "sample_submission.csv"
WORK_ROOT = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path(".").resolve()
ART_ROOT = WORK_ROOT / "exp517_stage22_pf1_artifacts"
ART_ROOT.mkdir(parents=True, exist_ok=True)

if hashlib.sha256(PUBLIC_CONFIG_JSON.encode("utf-8")).hexdigest() != PUBLIC_CONFIG_TEXT_SHA256:
    raise RuntimeError("embedded public v96 config SHA drift")
config_path = ART_ROOT / "pf_banks_config.json"
config_path.write_text(PUBLIC_CONFIG_JSON, encoding="utf-8")
empty_anchor_path = ART_ROOT / "empty_anchor.pkl"
empty_anchor_path.write_bytes(pickle.dumps({}, protocol=4))

gpu_count = torch.cuda.device_count()
gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
if gpu_count != 2 or not all("T4" in name for name in gpu_names):
    raise RuntimeError(f"fixed proxy run requires T4 x2, found {gpu_count}: {gpu_names}")

os.environ.update({
    "ROGII_DATA": str(DATA_ROOT),
    "ROGII_PROJ": str(WORK_ROOT),
    "ROGII_ART95": str(ART_ROOT),
    "V93_ANCHOR_PKL": str(empty_anchor_path),
    "PF_NGPU": "2",
    "PF_WELL_CHUNK": str(PF_WELL_CHUNK),
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
    "fidelity": FIDELITY,
    "data_root": str(DATA_ROOT),
    "sample_rows": int(len(sample)),
    "gpu_names": gpu_names,
    "public_kernel": PUBLIC_KERNEL,
})

# %% [markdown]
# ## 3. Public GPU particle-filter and fixed-lag smoother engine
# This is the released pf_banks_v95 numerical engine with only its terminal smoke block removed. The runtime path below forces its fixed-lag branch; the included whole-interval implementation is not executed.

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
# ## 4. Run pf_1 × twGR × fixed-lag 192 only
# Exactly one final-public pf_1 parameter set and one typewell-GR representation are allowed. Anchor, learned emission, self/neighbor representations, full smoothing, tabular fusion, and post-processing are disabled.

# %%
public_config = json.loads(config_path.read_text(encoding="utf-8"))
if public_config["bank_order"] != ["pf_1", "pf_2", "pf_3", "r0_seed32", "r1_seed32", "pfA"]:
    raise RuntimeError("unexpected public v96 bank order")
P = bank_param(PF_BANK)
P["smooth_mode"] = PF_SMOOTH_MODE
P["smooth_lag"] = PF_SMOOTH_LAG
P["_physics"] = PF_PHYSICS_ENABLED
P["_w_nn"] = PF_EMISSION_WEIGHT
P["_ps_combo_tau"] = 0.0
expected_contract = {
    "n_particles": PF_N_PARTICLES,
    "smooth_mode": PF_SMOOTH_MODE,
    "smooth_lag": PF_SMOOTH_LAG,
    "use_anchor": False,
    "use_phys": False,
    "_physics": False,
    "_w_nn": 0.0,
}
for key, expected in expected_contract.items():
    actual = P.get(key)
    if actual != expected:
        raise RuntimeError(f"pf_1 proxy contract drift for {key}: {actual!r} != {expected!r}")
if N_SEED != PF_N_SEEDS:
    raise RuntimeError(f"PF seed-count drift: {N_SEED} != {PF_N_SEEDS}")

sample_keys = sample["id"].astype(str)
parsed = sample_keys.str.rsplit("_", n=1, expand=True)
if parsed.shape[1] != 2:
    raise RuntimeError("sample ids must end in an underscore-separated row index")
sample_wells = parsed[0].astype(str)
try:
    parsed_row_ids = parsed[1].astype(int)
except Exception as exc:
    raise RuntimeError("sample id row suffix is not integer") from exc
if parsed_row_ids.isna().any():
    raise RuntimeError("sample id row suffix contains missing values")

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
    x = attach_anchor(x, wid, physics=False)
    if "_sim" in x or "_st" in x:
        raise RuntimeError(f"{wid} unexpectedly contains learned-emission inputs")
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
    w_nn=PF_EMISSION_WEIGHT,
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
    "smooth_lag": PF_SMOOTH_LAG,
    "physics": PF_PHYSICS_ENABLED,
    "emission_weight": PF_EMISSION_WEIGHT,
    "pf_seconds": pf_seconds,
})

# %% [markdown]
# ## 5. Sample-ID alignment, manifests, and LATE SUBMIT output
# The runtime sample submission is the sole schema/order authority. The direct fixed-lag pf_1 mean is written without fusion, gain, or visible-test branch, with logical candidate and submission SHA manifests.

# %%
candidate = pd.DataFrame({
    "id": sample_keys,
    "pf1_tw_fl192_mean": [prediction_lookup[key][0] for key in sample_keys],
    "pf1_tw_fl192_std": [prediction_lookup[key][1] for key in sample_keys],
    "pf1_tw_fl192_loglik": [prediction_lookup[key][2] for key in sample_keys],
})
submission = sample[["id"]].copy()
submission[TARGET_COLUMN] = candidate["pf1_tw_fl192_mean"].to_numpy(dtype=np.float64)

if len(submission) != len(sample) or not submission["id"].equals(sample["id"]):
    raise RuntimeError("submission id/order/row contract failed")
if submission["id"].astype(str).duplicated().any():
    raise RuntimeError("submission contains duplicate ids")
if not np.isfinite(submission[TARGET_COLUMN].to_numpy(dtype=np.float64)).all():
    raise RuntimeError("submission contains non-finite predictions")

candidate_content = candidate.to_csv(index=False, lineterminator="\n")
candidate_gzip_path = WORK_ROOT / "pf1_tw_fl192_candidate.csv.gz"
with gzip.open(candidate_gzip_path, "wt", encoding="utf-8", newline="") as handle:
    handle.write(candidate_content)
submission_path = WORK_ROOT / "submission.csv"
submission.to_csv(submission_path, index=False, lineterminator="\n")

component_manifest = {
    "experiment": EXPERIMENT,
    "fidelity": FIDELITY,
    "learned_models": [],
    "pf": {
        "bank": PF_BANK,
        "representation": PF_REPRESENTATION,
        "public_config_raw_sha256": PUBLIC_CONFIG_SHA256,
        "runtime_config_text_sha256": sha256_path(config_path),
        "particles": PF_N_PARTICLES,
        "seeds": PF_N_SEEDS,
        "smooth_mode": PF_SMOOTH_MODE,
        "smooth_lag": PF_SMOOTH_LAG,
        "physics": PF_PHYSICS_ENABLED,
        "emission_weight": PF_EMISSION_WEIGHT,
        "generation_seed": PF_GENERATION_SEED,
    },
}
(WORK_ROOT / "component_manifest.json").write_text(
    json.dumps(component_manifest, indent=2, sort_keys=True), encoding="utf-8"
)

manifest = {
    "experiment": EXPERIMENT,
    "fidelity": FIDELITY,
    "late_submission": True,
    "late_submission_phase": LATE_SUBMISSION_PHASE,
    "submission_message": "LATE SUBMIT | exp517 | stage2-2 pf1 x twGR fixedlag192 proxy | fixed v1",
    "source": {
        "public_kernel": PUBLIC_KERNEL,
        "public_kernel_id_no": PUBLIC_KERNEL_ID_NO,
        "public_notebook_sha256": PUBLIC_NOTEBOOK_SHA256,
        "public_pf_source_sha256": PUBLIC_PF_SOURCE_SHA256,
        "public_config_raw_sha256": PUBLIC_CONFIG_SHA256,
        "public_config_embedded_text_sha256": PUBLIC_CONFIG_TEXT_SHA256,
    },
    "runtime": {
        "gpu_names": gpu_names,
        "pf_seconds": pf_seconds,
        "well_count": len(names),
        "row_count": len(candidate),
    },
    "artifacts": {
        "candidate_decompressed_content_sha256": sha256_text(candidate_content),
        "candidate_raw_gzip_sha256": sha256_path(candidate_gzip_path),
        "prediction_content_sha256": logical_csv_sha(candidate[["id", "pf1_tw_fl192_mean"]]),
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
    "published_stage22_system_reference": {
        "cv": 7.50,
        "public": 6.724,
        "private": 7.404,
        "comparable_to_this_pf_only_proxy": False,
        "reason": "published result used five PF inputs plus a tabular model",
    },
    "deterministic_anchor": False,
    "deterministic_reason": "GPU fixed-environment rerun equality not tested",
}
(WORK_ROOT / "exp517_execution_manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
)
print(json.dumps({
    "LATE_SUBMIT": True,
    "fidelity": FIDELITY,
    "submission": str(submission_path),
    "submission_sha256": manifest["artifacts"]["submission_sha256"],
    "candidate_content_sha256": manifest["artifacts"]["candidate_decompressed_content_sha256"],
    "contract": manifest["submission_contract"],
}, indent=2))

