# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
#   language_info:
#     name: python
# jupytext:
#   text_representation:
#     extension: .py
#     format_name: percent
#     format_version: '1.3'
# ---

# %% [markdown]
# # exp517 stage 2-2 five-PF fixed-lag-192 tabular train — corrected v2
#
# v1の1-PF direct提出は契約不一致の失敗履歴として保持する。
# この候補は同じexp517内で、公開writeupの5 PF + smoother + tabular契約を実装する。

# %% [markdown]
# ## 1. Method contract
# input=Ravaghi public feature frame + five original Optuna PF trajectories; target=TVT-last_known_tvt; output=row residual; loss=GBDT RMSE + Ridge; decode=public PF blend/fade/SG; context=well suffix fixed-lag192 + row tabular.

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
IMPLEMENTATION_VERSION = "stage22_corrected_v2"
FIDELITY = "historical_contract_reconstruction"
PUBLIC_KERNEL = "k256net/public20th-private6th-pf-pf-pf-pf-and-bagging"
PUBLIC_KERNEL_ID_NO = 126919690
PUBLIC_NOTEBOOK_SHA256 = "b44f7889d6abdf9b027d33cb6c6b45f23902d609fae8d06f332914017784c924"
PUBLIC_CONFIG_SHA256 = "80e973d5f5e0e39be758a03f399cdd3d81d9e79320da8db6fbddbc25c2a202f3"
PUBLIC_CONFIG_TEXT_SHA256 = "aff2bcf63d1dd0b24ceefcd77a8b4fc058e0977e4e9518b58c6fea8c5468d962"
PUBLIC_PF_SOURCE_SHA256 = "ea5e5af2a6fe6e344ad3a792c2735368dbe9f2f61aae79b804abe3eb493e6a6e"
PUBLIC_TABULAR_SOURCE_SHA256 = "35c2bc435f5989eb1811d9c57cfec2b3f93c143353fae931a4431f77e53b691f"
PF_BANKS = ['pf_1', 'pf_2', 'pf_3', 'r0_seed32', 'r1_seed32']
PF_REPRESENTATION = "tw"
PF_GENERATION_SEED = 4423098
PF_N_SEEDS = 32
PF_SMOOTH_MODE = "fixedlag"
PF_SMOOTH_LAG = 192
PF_WELL_CHUNK = 40
PF_OFFSETS = np.array([-30, -15, -8, -4, -2, 0, 2, 4, 8, 15, 30], dtype=np.float32)
PUBLIC_CONFIG_JSON = '{\n "bank_order": [\n  "pf_1",\n  "pf_2",\n  "pf_3",\n  "r0_seed32",\n  "r1_seed32",\n  "pfA"\n ],\n "physics_banks": [\n  "pfA"\n ],\n "w_nn_bank": {\n  "pfA": 0.01\n },\n "w_nn_default": 0.02,\n "n_seed": 32,\n "smooth_lag": 192,\n "anchor_pkl": "grfree_anchor_train.pkl",\n "params": {\n  "pf_1": {\n   "gr_power": 2.1947763195255123,\n   "gr_sig_def": 114.11439615641156,\n   "gr_sig_max": 32.79346108797473,\n   "gr_sig_min": 7.495879205722816,\n   "gr_sig_mult": 2.7482077023565075,\n   "hgr_smooth_r": 3,\n   "init_pos_std": 0.43824435375436277,\n   "init_rate_std": 0.001840222365629066,\n   "jump_prob": 0.002,\n   "jump_std": 5.0,\n   "likelihood_scale": 20.0,\n   "mom": 0.998833416480357,\n   "n_particles": 600,\n   "pn": 0.006788762926604484,\n   "rate_clip": 2.0,\n   "resamp": 0.060642973044944336,\n   "rp": 0.6864944651997384,\n   "rr": 0.00015873788792956448,\n   "tvt_clip_margin": 113.08092552242186,\n   "tw_gr_smooth_r": 0,\n   "vn": 0.0013871256161405515,\n   "phys_sig": 0.05095402721954007,\n   "use_phys": false,\n   "smooth_lag": 192,\n   "use_anchor": false\n  },\n  "pf_2": {\n   "gr_sig_def": 71.37105679394953,\n   "gr_sig_max": 44.7373292550486,\n   "gr_sig_min": 6.520200142977573,\n   "gr_sig_mult": 1.9000654781073127,\n   "init_pos_std": 1.1039311766848976,\n   "init_rate_std": 0.009213697911348712,\n   "likelihood_scale": 18.584958539746772,\n   "mom": 0.9994265753132063,\n   "n_particles": 600,\n   "pn": 0.011257078147676399,\n   "resamp": 0.30834451379977984,\n   "rp": 0.49711567429512854,\n   "rr": 0.00031680690540158854,\n   "tvt_clip_margin": 63.752292532272826,\n   "vn": 0.0010650490363481595,\n   "rate_clip": 0.0,\n   "gr_power": 2.0,\n   "hgr_smooth_r": 0,\n   "tw_gr_smooth_r": 0,\n   "jump_prob": 0.0,\n   "jump_std": 10.0,\n   "phys_sig": 0.05095402721954007,\n   "use_phys": false,\n   "smooth_lag": 192,\n   "use_anchor": false\n  },\n  "pf_3": {\n   "gr_power": 2.074149546524975,\n   "gr_sig_def": 66.97913258656452,\n   "gr_sig_max": 50.843656140023995,\n   "gr_sig_min": 9.427442041004895,\n   "gr_sig_mult": 2.4485083695548218,\n   "hgr_smooth_r": 0,\n   "init_pos_std": 2.3279966476603975,\n   "init_rate_std": 0.03938758391370206,\n   "jump_prob": 0.002,\n   "jump_std": 5.0,\n   "likelihood_scale": 14.890876838217093,\n   "mom": 0.9999769474035675,\n   "n_particles": 600,\n   "pn": 0.04394815893193551,\n   "rate_clip": 0.0,\n   "resamp": 0.054606961294962585,\n   "rp": 0.42322457711970995,\n   "rr": 0.002148810696831248,\n   "tvt_clip_margin": 91.4794941498565,\n   "tw_gr_smooth_r": 0,\n   "vn": 0.0006883646035942229,\n   "phys_sig": 0.05095402721954007,\n   "use_phys": false,\n   "smooth_lag": 192,\n   "use_anchor": false,\n   "robust_nu": 5.0,\n   "temper_beta": 0.85\n  },\n  "r0_seed32": {\n   "n_particles": 400,\n   "likelihood_scale": 10.536319667317654,\n   "init_pos_std": 0.9331541486020234,\n   "init_rate_std": 0.00047753793163756304,\n   "mom": 0.9997911117425017,\n   "vn": 0.0006297672017716754,\n   "pn": 0.005014009140727263,\n   "rp": 0.7627309890952053,\n   "rr": 0.00045663430474523125,\n   "resamp": 0.032049849143995945,\n   "gr_sig_min": 11.865793233968713,\n   "gr_sig_max": 44.98927964612316,\n   "gr_sig_def": 119.25397464636981,\n   "gr_sig_mult": 3.058318309943017,\n   "tvt_clip_margin": 150.56111180234504,\n   "rate_clip": 2.0,\n   "gr_power": 1.7826896851395453,\n   "hgr_smooth_r": 0,\n   "tw_gr_smooth_r": 0,\n   "jump_prob": 0.002,\n   "jump_std": 5.0,\n   "phys_sig": 0.013666550145707291,\n   "use_phys": false,\n   "smooth_lag": 192,\n   "use_anchor": false,\n   "robust_nu": 5.0,\n   "temper_beta": 1.0\n  },\n  "r1_seed32": {\n   "n_particles": 400,\n   "likelihood_scale": 43.06423939744663,\n   "init_pos_std": 1.472927959941933,\n   "init_rate_std": 0.0006649111513262288,\n   "mom": 0.9995184091067627,\n   "vn": 0.0012160450940772792,\n   "pn": 0.002809741394025188,\n   "rp": 0.7070613341717283,\n   "rr": 0.0007812869001227403,\n   "resamp": 0.036600490411087865,\n   "gr_sig_min": 6.324062861615175,\n   "gr_sig_max": 29.32935945549601,\n   "gr_sig_def": 83.2600219362552,\n   "gr_sig_mult": 1.7565330745434495,\n   "tvt_clip_margin": 177.98411545472226,\n   "rate_clip": 2.5,\n   "gr_power": 1.4643521376101742,\n   "hgr_smooth_r": 0,\n   "tw_gr_smooth_r": 0,\n   "jump_prob": 0.002,\n   "jump_std": 10.0,\n   "phys_sig": 0.010953433688074062,\n   "use_phys": false,\n   "smooth_lag": 192,\n   "use_anchor": false,\n   "robust_nu": 5.0,\n   "temper_beta": 0.85\n  },\n  "pfA": {\n   "n_particles": 600,\n   "likelihood_scale": 29.056396339290437,\n   "init_pos_std": 1.6168048868046325,\n   "init_rate_std": 0.014412715870661067,\n   "mom": 0.9995682504305502,\n   "vn": 0.000873883338794689,\n   "pn": 0.00606209538025288,\n   "rp": 0.14025726060525087,\n   "rr": 0.00398567931602027,\n   "resamp": 0.18645022443117387,\n   "gr_sig_min": 10.63834771185878,\n   "gr_sig_max": 43.95858017409127,\n   "gr_sig_def": 86.87561008087175,\n   "gr_sig_mult": 2.7681905590609426,\n   "tvt_clip_margin": 86.30698497112195,\n   "rate_clip": 0.0,\n   "gr_power": 1.85175446220882,\n   "hgr_smooth_r": 2,\n   "tw_gr_smooth_r": 0,\n   "jump_prob": 0.002,\n   "jump_std": 5.0,\n   "phys_sig": 2.3072090423765363,\n   "use_anchor": true,\n   "anchor_mult": 20.0,\n   "anchor_ramp_md": 0.0,\n   "lk_floor": 1e-05,\n   "gr_debias": 0,\n   "rate_target": "zero",\n   "grid_step": 0.2,\n   "self_mix_w": 1.0,\n   "nbr_mix_w": 1.0,\n   "use_phys": false,\n   "smooth_lag": 192,\n   "robust_nu": 5.0,\n   "temper_beta": 1.0\n  }\n },\n "smooth_mode": "full"\n}'
STAGE22_PF_CACHE: dict[str, dict[str, dict[str, object]]] = {}


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def resolve_competition_root() -> Path:
    roots = [
        Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction"),
        Path("/kaggle/input/rogii-wellbore-geology-prediction"),
        Path("data/raw").resolve(),
    ]
    if Path("/kaggle/input").is_dir():
        roots.extend(path.parent for path in Path("/kaggle/input").rglob("sample_submission.csv"))
    for root in roots:
        if (root / "train").is_dir() and (root / "test").is_dir() and (root / "sample_submission.csv").is_file():
            return root
    raise FileNotFoundError("competition root with train/test/sample_submission.csv was not found")


DATA_ROOT = resolve_competition_root()
WORK_ROOT = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path(".").resolve()
ART_ROOT = WORK_ROOT / "exp517_stage22_v2_runtime"
ART_ROOT.mkdir(parents=True, exist_ok=True)
if sha256_text(PUBLIC_CONFIG_JSON) != PUBLIC_CONFIG_TEXT_SHA256:
    raise RuntimeError("embedded public config text SHA drift")
config_path = ART_ROOT / "pf_banks_config.json"
config_path.write_text(PUBLIC_CONFIG_JSON, encoding="utf-8")
empty_anchor_path = ART_ROOT / "empty_anchor.pkl"
empty_anchor_path.write_bytes(pickle.dumps({}, protocol=4))

gpu_count = torch.cuda.device_count()
gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
if gpu_count < 1:
    raise RuntimeError("stage2-2 corrected run requires a Kaggle GPU")
os.environ.update({
    "ROGII_DATA": str(DATA_ROOT),
    "ROGII_OUT": str(WORK_ROOT),
    "ROGII_PROJ": str(WORK_ROOT),
    "ROGII_ART95": str(ART_ROOT),
    "V93_ANCHOR_PKL": str(empty_anchor_path),
    "PF_NGPU": str(gpu_count),
    "PF_WELL_CHUNK": str(PF_WELL_CHUNK),
    "PS_COMBO_TAU": "0",
    "USE_GPU": "gpu",
    "PYTHONUNBUFFERED": "1",
})

# %% [markdown]
# ## 2. Released GPU PF engine
# The released source is embedded and SHA-pinned. Runtime overrides disable all post-stage2 likelihood/anchor/emission changes.

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
# ## 3. Public tabular feature engine
# The strict public Ravaghi replay is embedded. Only its base PF call is redirected to the corrected pf_1 fixed-lag cache.

# %%
# ruff: noqa

import json
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

def run_pf_ancc(hw, tw_tvt, tw_gr, N=ANCC_N):
    gs = _gr_sig(hw, tw_tvt, tw_gr); kn = hw[hw.TVT_input.notna()]; ev = hw[hw.TVT_input.isna()]
    if len(ev) == 0: return np.array([]), np.array([])
    ls = float(kn.TVT_input.iloc[-1]+kn.Z.iloc[-1])
    tail = kn.tail(30); dt = np.diff(tail.TVT_input.values); dz = np.diff(tail.Z.values); dm = np.diff(tail.MD.values); m = dm > 0
    ir = float(np.median((dt+dz)[m]/dm[m])) if m.sum() >= 3 else 0.
    gg, gmin, gst = _grid(tw_tvt, tw_gr)
    pts, std = _pf_ancc(ev.MD.values.astype(np.float64), ev.Z.values.astype(np.float64), ev.GR.values.astype(np.float64),
                        gg, gmin, gst, gs, ls, ir, N, ANCC_ALPHA, ANCC_RN, ANCC_PN, ANCC_IS, ANCC_RP, ANCC_RR, PF_RESAMP)
    return pts.astype(np.float32), std.astype(np.float32)

def run_pf_z(hw, tw_tvt, tw_gr, N=PF_N):
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
    pts, std = _pf_z(ev.MD.values.astype(np.float64), ev.Z.values.astype(np.float64), ev.GR.values.astype(np.float64),
                     gr_sm.loc[ev.index].values.astype(np.float64), gg, gs2, gmin, gst, gs,
                     float(kna.TVT_input.iloc[-1]), iv, beta, icpt, zsig, N,
                     PF_MOM, PF_VN, PF_PN, PF_GR_WT, PF_ROUGH_P, PF_ROUGH_V, PF_RESAMP)
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
_beam_jit(np.random.randn(30), np.random.randn(50), 25, 8, 15., 100.)
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
    _stage22 = STAGE22_PF_CACHE[wid]["pf_1"]
    pf_a = np.asarray(_stage22["mean"], dtype=np.float32)
    std_a = np.asarray(_stage22["std"], dtype=np.float32)
    if len(pf_a) == 0: return None
    pf_z, std_z = run_pf_z(hw, tw_tvt, tw_gr)
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
    out, idx, _ = lik_pf(hw, tw)
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

# %% [markdown]
# ## 4. Five-PF feature and decode contract
# All five banks are generated; the public one-PF feature family is repeated per bank and pf_1 remains the unsuffixed compatibility alias.

# %%
def stage22_bank_param(bank: str) -> dict:
    if bank not in PF_BANKS:
        raise KeyError(bank)
    p = bank_param(bank)
    p["smooth_mode"] = PF_SMOOTH_MODE
    p["smooth_lag"] = PF_SMOOTH_LAG
    p["use_anchor"] = False
    p["use_phys"] = False
    p["robust_nu"] = 0.0
    p["temper_beta"] = 1.0
    p["_physics"] = False
    p["_w_nn"] = 0.0
    p["_ps_combo_tau"] = 0.0
    return p


def list_wells(split: str) -> list[str]:
    return sorted(path.name.split("__", 1)[0] for path in (DATA_ROOT / split).glob("*__horizontal_well.csv"))


def generate_stage22_pf_cache(split: str, wells: list[str]) -> tuple[pd.DataFrame, dict]:
    STAGE22_PF_CACHE.clear()
    meta: dict[str, dict[str, object]] = {}
    for wid in wells:
        hw = pd.read_csv(DATA_ROOT / split / f"{wid}__horizontal_well.csv")
        tw = pd.read_csv(DATA_ROOT / split / f"{wid}__typewell.csv").sort_values("TVT")
        rows = np.flatnonzero(hw["TVT_input"].isna().to_numpy())
        if not len(rows):
            continue
        known = hw.loc[hw["TVT_input"].notna(), "TVT_input"]
        if len(known) < 10:
            continue
        gr = hw["GR"].astype(float).interpolate(limit_direction="both").fillna(float(np.nanmean(tw["GR"]))).to_numpy()
        meta[wid] = {
            "rows": rows,
            "last_known_tvt": float(known.iloc[-1]),
            "horizontal_gr": gr[rows].astype(np.float32),
            "tw_tvt": tw["TVT"].to_numpy(np.float64),
            "tw_gr": tw["GR"].to_numpy(np.float64),
        }
        STAGE22_PF_CACHE[wid] = {}

    t0 = time.perf_counter()
    bank_seconds: dict[str, float] = {}
    for bank in PF_BANKS:
        p = stage22_bank_param(bank)
        names: list[str] = []
        inps: list[dict] = []
        for wid in meta:
            hw = pd.read_csv(DATA_ROOT / split / f"{wid}__horizontal_well.csv")
            tw = pd.read_csv(DATA_ROOT / split / f"{wid}__typewell.csv").sort_values("TVT")
            x = build_smoother_inputs(hw, tw["TVT"].to_numpy(float), tw["GR"].to_numpy(float), p)
            if x is None:
                raise RuntimeError(f"{wid}/{bank}: no PF suffix")
            x = attach_anchor(x, wid, physics=False)
            x.pop("_sim", None); x.pop("_st", None)
            names.append(wid); inps.append(x)
        started = time.perf_counter()
        outputs = run_smoother_ext(
            inps, p, seed=PF_GENERATION_SEED, n_seeds=PF_N_SEEDS,
            chunk=PF_WELL_CHUNK, w_nn=0.0,
        )
        bank_seconds[bank] = time.perf_counter() - started
        if len(outputs) != len(names) or any(value is None for value in outputs):
            raise RuntimeError(f"{bank}: incomplete PF outputs")
        for wid, output in zip(names, outputs):
            mean = np.asarray(output["mean"], dtype=np.float32)
            std = np.asarray(output["std"], dtype=np.float32)
            if len(mean) != len(meta[wid]["rows"]) or not np.isfinite(mean).all() or not np.isfinite(std).all():
                raise RuntimeError(f"{wid}/{bank}: invalid PF output")
            STAGE22_PF_CACHE[wid][bank] = {"mean": mean, "std": std, "loglik": float(output["loglik"])}
        print({"bank": bank, "seconds": bank_seconds[bank], "wells": len(names)}, flush=True)

    parts: list[pd.DataFrame] = []
    for wid, info in meta.items():
        rows = np.asarray(info["rows"], dtype=np.int64)
        last = float(info["last_known_tvt"])
        hgr = np.asarray(info["horizontal_gr"], dtype=np.float32)
        tw_tvt = np.asarray(info["tw_tvt"], dtype=np.float64)
        tw_gr = np.asarray(info["tw_gr"], dtype=np.float64)
        values: dict[str, object] = {
            "id": [f"{wid}_{int(row)}" for row in rows],
            "well": wid,
        }
        for idx, bank in enumerate(PF_BANKS, start=1):
            output = STAGE22_PF_CACHE[wid][bank]
            mean = np.asarray(output["mean"], dtype=np.float32)
            values[f"pf_ancc_{idx}"] = mean
            values[f"pf_ancc_std_{idx}"] = np.asarray(output["std"], dtype=np.float32)
            values[f"pf_ancc_delta_{idx}"] = (mean - np.float32(last)).astype(np.float32)
            for offset in PF_OFFSETS:
                values[f"tdpf{int(offset)}_{idx}"] = (
                    hgr - np.interp(mean + float(offset), tw_tvt, tw_gr).astype(np.float32)
                ).astype(np.float32)
        parts.append(pd.DataFrame(values))
    frame = pd.concat(parts, ignore_index=True)
    if frame["id"].duplicated().any():
        raise RuntimeError("PF feature frame contains duplicate ids")
    manifest = {
        "split": split,
        "banks": PF_BANKS,
        "wells": int(frame["well"].nunique()),
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "bank_seconds": bank_seconds,
        "total_seconds": time.perf_counter() - t0,
    }
    return frame, manifest


def augment_stage22_frame(base: pd.DataFrame, pf_frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    if len(base) != len(pf_frame) or not base["id"].astype(str).equals(pf_frame["id"].astype(str)):
        pf_frame = base[["id"]].merge(pf_frame, on="id", how="left", validate="one_to_one")
    if pf_frame.drop(columns=["id", "well"], errors="ignore").isna().any().any():
        raise RuntimeError("PF feature alignment produced missing values")
    # The public train CSV is 7.4 GB on disk. Mutate the loaded frame in place so
    # the corrected run does not hold a second full public frame in RAM.
    out = base
    last = out["last_known_tvt"].to_numpy(np.float32)
    spatial = out["tvtF_ANCC"].to_numpy(np.float32)
    dense = last + out["tvt_dense_d"].to_numpy(np.float32)
    pf_z = out["pf_z"].to_numpy(np.float32)
    added: dict[str, np.ndarray] = {}
    for idx, bank in enumerate(PF_BANKS, start=1):
        mean = pf_frame[f"pf_ancc_{idx}"].to_numpy(np.float32)
        for column in [f"pf_ancc_{idx}", f"pf_ancc_std_{idx}", f"pf_ancc_delta_{idx}"]:
            added[column] = pf_frame[column].to_numpy(np.float32)
        added[f"pf_vs_z_{idx}"] = (mean - pf_z).astype(np.float32)
        added[f"pf_vs_spatial_{idx}"] = (mean - spatial).astype(np.float32)
        added[f"pf_vs_dense_{idx}"] = (mean - dense).astype(np.float32)
        for offset in PF_OFFSETS:
            column = f"tdpf{int(offset)}_{idx}"
            added[column] = pf_frame[column].to_numpy(np.float32)
    overlap = sorted(set(added).intersection(out.columns))
    if overlap:
        raise RuntimeError(f"stage2 suffixed feature collision: {overlap[:20]}")
    out = pd.concat([out, pd.DataFrame(added, index=out.index)], axis=1)

    out["pf_ancc"] = out["pf_ancc_1"].to_numpy(np.float32)
    out["pf_ancc_std"] = out["pf_ancc_std_1"].to_numpy(np.float32)
    out["pf_ancc_delta"] = out["pf_ancc_delta_1"].to_numpy(np.float32)
    out["pf_vs_z"] = out["pf_vs_z_1"].to_numpy(np.float32)
    out["pf_vs_spatial"] = out["pf_vs_spatial_1"].to_numpy(np.float32)
    out["pf_vs_dense"] = out["pf_vs_dense_1"].to_numpy(np.float32)
    for offset in PF_OFFSETS:
        out[f"tdpf{int(offset)}"] = out[f"tdpf{int(offset)}_1"].to_numpy(np.float32)

    candidate_paths = [out[f"pf_ancc_{idx}"].to_numpy(np.float32) for idx in range(1, 6)]
    signal_paths = candidate_paths + [
        last + out[f"beam_{tag}_d"].to_numpy(np.float32) for *_, tag in BEAMS
    ] + [
        last + out["sc8_d"].to_numpy(np.float32),
        last + out["sc15_d"].to_numpy(np.float32),
        last + out["sc25_d"].to_numpy(np.float32),
        last + out["sc_ens_d"].to_numpy(np.float32),
        spatial,
        dense,
    ]
    signal_matrix = np.stack(signal_paths, axis=1)
    out["sig_std"] = signal_matrix.std(axis=1).astype(np.float32)
    out["sig_mean_d"] = (signal_matrix.mean(axis=1) - last).astype(np.float32)

    excluded = {"id", "well", "target"}
    features = [column for column in out.columns if column not in excluded]
    for column in features:
        out[column] = pd.to_numeric(out[column], errors="coerce").astype(np.float32)
    nonfinite = [column for column in features if not np.isfinite(out[column].to_numpy(np.float32)).all()]
    if nonfinite:
        raise RuntimeError(f"stage2 feature frame contains non-finite columns: {nonfinite[:20]}")
    return out, features


def apply_public_postprocess(df: pd.DataFrame, model_delta: np.ndarray, pf_delta: np.ndarray) -> np.ndarray:
    delta = np.asarray(model_delta, float) * 0.91 + np.asarray(pf_delta, float) * 0.09
    delta *= 1.0 - np.exp(-np.maximum(df["md_since"].to_numpy(float), 0.0) / 85.0)
    return delta


def sg_smooth_by_well(df: pd.DataFrame, values: np.ndarray, window: int = 17, poly: int = 3) -> np.ndarray:
    result = np.asarray(values, float).copy()
    for _, group in df.groupby("well", sort=False):
        idx = group.index.to_numpy()
        width = min(window, len(idx))
        if width % 2 == 0:
            width -= 1
        if width >= poly + 2:
            result[idx] = savgol_filter(result[idx], width, poly)
    return result

# %% [markdown]
# ## 5. Orchestration
# Train only the corrected scientific variant.

# %%
def resolve_public_train_csv() -> Path:
    candidates = [
        Path("/kaggle/input/wellbore-geology-prediction-artifacts/data/train.csv"),
        Path("/kaggle/input/datasets/ravaghi/wellbore-geology-prediction-artifacts/data/train.csv"),
    ]
    if Path("/kaggle/input").is_dir():
        candidates.extend(Path("/kaggle/input").glob("**/wellbore-geology-prediction-artifacts/data/train.csv"))
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError("Ravaghi public artifact data/train.csv was not found")


from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.linear_model import Ridge

train_wells = list_wells("train")
pf_frame, pf_runtime = generate_stage22_pf_cache("train", train_wells)
STAGE22_PF_CACHE.clear()
public_train_path = resolve_public_train_csv()
print({"public_train_csv": str(public_train_path), "bytes": public_train_path.stat().st_size}, flush=True)
train_df = pd.read_csv(public_train_path, low_memory=False)
train_df, features = augment_stage22_frame(train_df, pf_frame)
del pf_frame

expected_rows = 3_783_989
expected_wells = 773
if len(train_df) != expected_rows or train_df["well"].nunique() != expected_wells:
    raise RuntimeError(f"public train coverage drift: rows={len(train_df)} wells={train_df['well'].nunique()}")

model_dir = WORK_ROOT / "exp517_stage22_v2_model"
model_dir.mkdir(parents=True, exist_ok=True)
X = train_df[features].to_numpy(np.float32)
y = train_df["target"].to_numpy(np.float32)
groups = train_df["well"].astype(str).to_numpy()
ids = train_df["id"].astype(str).to_numpy()
base = train_df["last_known_tvt"].to_numpy(np.float32)
pf_delta = train_df["pf_ancc_delta_1"].to_numpy(np.float32)

lgb_params = [
    dict(boosting_type="gbdt", num_leaves=255, min_child_samples=15, subsample=0.8, subsample_freq=1,
         colsample_bytree=0.8, reg_lambda=3.0, reg_alpha=0.05, objective="regression", verbose=-1,
         n_jobs=-1, device_type="gpu", gpu_use_dp=False, max_bin=255, learning_rate=0.03,
         n_estimators=5000, seed=123),
    dict(n_jobs=-1, verbose=-1, reg_alpha=10.788188919840913, subsample=0.47437582748953966,
         num_leaves=64, reg_lambda=95.75401894533888, n_estimators=10000, random_state=0,
         boosting_type="gbdt", learning_rate=0.00934485794382918,
         colsample_bytree=0.39283351290380497, min_child_weight=0.24081152127177283,
         min_child_samples=40, device="gpu"),
    dict(n_jobs=-1, verbose=-1, reg_alpha=10.788188919840913, subsample=0.47437582748953966,
         num_leaves=64, reg_lambda=95.75401894533888, n_estimators=10000, random_state=29,
         boosting_type="gbdt", learning_rate=0.00934485794382918,
         colsample_bytree=0.39283351290380497, min_child_weight=0.24081152127177283,
         min_child_samples=40, device="gpu"),
]
cb_params = [
    dict(iterations=8000, depth=7, l2_leaf_reg=2.0, min_data_in_leaf=15, border_count=254,
         loss_function="RMSE", task_type="GPU", devices="0", od_type="Iter", od_wait=300,
         verbose=0, learning_rate=0.02, random_seed=7),
    dict(iterations=8000, depth=7, l2_leaf_reg=2.0, min_data_in_leaf=15, border_count=254,
         loss_function="RMSE", task_type="GPU", devices="0", od_type="Iter", od_wait=300,
         verbose=0, learning_rate=0.03, random_seed=123),
]

cv = GroupKFold(n_splits=5)
splits = list(cv.split(X, y, groups=groups))
base_oof = np.zeros((len(train_df), 5), dtype=np.float32)
model_records: list[dict] = []

for config_index, params in enumerate(lgb_params):
    for fold, (tr, va) in enumerate(splits):
        model = LGBMRegressor(**params)
        model.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], eval_metric="rmse",
                  callbacks=[early_stopping(250, verbose=False), log_evaluation(0)])
        best = int(model.best_iteration_ or params["n_estimators"])
        base_oof[va, config_index] = model.predict(X[va], num_iteration=best).astype(np.float32)
        filename = f"lightgbm_{config_index}_fold{fold}.txt"
        model.booster_.save_model(str(model_dir / filename), num_iteration=best)
        model_records.append({"family": "lightgbm", "config": config_index, "fold": fold,
                              "best_iteration": best, "file": filename})
        print({"family": "lightgbm", "config": config_index, "fold": fold,
               "rmse": rmse(y[va], base_oof[va, config_index]), "best_iteration": best}, flush=True)

for config_index, params in enumerate(cb_params):
    column = 3 + config_index
    for fold, (tr, va) in enumerate(splits):
        model = CatBoostRegressor(**params)
        model.fit(X[tr], y[tr], eval_set=(X[va], y[va]), early_stopping_rounds=250, use_best_model=True)
        base_oof[va, column] = model.predict(X[va]).astype(np.float32)
        filename = f"catboost_{config_index}_fold{fold}.cbm"
        model.save_model(str(model_dir / filename))
        model_records.append({"family": "catboost", "config": config_index, "fold": fold,
                              "best_iteration": int(model.get_best_iteration()), "file": filename})
        print({"family": "catboost", "config": config_index, "fold": fold,
               "rmse": rmse(y[va], base_oof[va, column]),
               "best_iteration": int(model.get_best_iteration())}, flush=True)

ridge_oof = np.zeros(len(train_df), dtype=np.float32)
ridge_records: list[dict] = []
for fold, (tr, va) in enumerate(splits):
    ridge = Ridge(alpha=1.6602834637650032, tol=0.0005030247295617308,
                  positive=True, fit_intercept=True, random_state=42)
    ridge.fit(base_oof[tr], y[tr])
    ridge_oof[va] = ridge.predict(base_oof[va]).astype(np.float32)
    filename = f"ridge_fold{fold}.npz"
    np.savez(model_dir / filename, coef=ridge.coef_.astype(np.float64), intercept=np.float64(ridge.intercept_))
    ridge_records.append({"fold": fold, "file": filename})

pp_delta = apply_public_postprocess(train_df, ridge_oof, pf_delta)
pred_tvt = base.astype(np.float64) + pp_delta
true_tvt = base.astype(np.float64) + y.astype(np.float64)
cv_ridge = rmse(y, ridge_oof)
cv_pp = rmse(true_tvt, pred_tvt)
cv_pp_sg = rmse(true_tvt, sg_smooth_by_well(train_df, pred_tvt))
fold_metrics = []
for fold, (_, va) in enumerate(splits):
    fold_metrics.append({"fold": fold, "rows": int(len(va)),
                         "ridge_rmse": rmse(y[va], ridge_oof[va]),
                         "postprocess_rmse": rmse(true_tvt[va], pred_tvt[va])})

oof_path = WORK_ROOT / "exp517_stage22_v2_oof.csv.gz"
pd.DataFrame({"id": ids, "well": groups, "target_tvt": true_tvt,
              "last_known_tvt": base, "pred_delta_ridge": ridge_oof,
              "pred_tvt_postprocess": pred_tvt}).to_csv(oof_path, index=False, compression="gzip")

for record in model_records:
    record["sha256"] = sha256_path(model_dir / record["file"])
for record in ridge_records:
    record["sha256"] = sha256_path(model_dir / record["file"])
manifest = {
    "experiment": EXPERIMENT,
    "implementation_version": IMPLEMENTATION_VERSION,
    "fidelity": FIDELITY,
    "method_contract": {
        "input": "Ravaghi public base frame + five original-Optuna twGR PF trajectories",
        "target": "TVT - last_known_tvt",
        "output": "row residual",
        "loss": "LightGBM/CatBoost RMSE + positive Ridge stack",
        "decode": "0.91 ridge + 0.09 pf_1; tau85 fade; SG17/3 at inference",
        "context_unit": "one well suffix; PF fixed-lag 192; row tabular; GroupKFold by well",
    },
    "source": {"public_notebook_sha256": PUBLIC_NOTEBOOK_SHA256,
               "public_config_sha256": PUBLIC_CONFIG_SHA256,
               "public_pf_source_sha256": PUBLIC_PF_SOURCE_SHA256,
               "public_tabular_source_sha256": PUBLIC_TABULAR_SOURCE_SHA256,
               "public_train_csv": str(public_train_path),
               "public_train_csv_sha256": sha256_path(public_train_path)},
    "features": features,
    "feature_count": len(features),
    "rows": len(train_df),
    "wells": int(train_df["well"].nunique()),
    "pf_runtime": pf_runtime,
    "lgb_params": lgb_params,
    "cb_params": cb_params,
    "models": model_records,
    "ridge_models": ridge_records,
    "execution_count": {"scientific_variants": 1, "pf_banks": 5, "representations": 1,
                        "lightgbm_configs": 3, "catboost_configs": 2, "folds": 5,
                        "base_models": 25, "ridge_models": 5, "control_reruns": 0},
    "metrics": {"ridge_rmse": cv_ridge, "postprocess_rmse": cv_pp,
                "postprocess_sg_diagnostic_rmse": cv_pp_sg, "published_stage22_cv": 7.50,
                "absolute_delta_from_published": abs(cv_pp - 7.50), "folds": fold_metrics},
    "oof": {"file": oof_path.name, "sha256": sha256_path(oof_path)},
    "gpu_names": gpu_names,
}
(model_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
(WORK_ROOT / "exp517_stage22_v2_metrics.json").write_text(
    json.dumps(manifest["metrics"], indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps({"stage22_v2": True, "cv_postprocess": cv_pp,
                  "published_cv": 7.50, "model_count": len(model_records),
                  "ridge_count": len(ridge_records), "feature_count": len(features)}, indent=2))

