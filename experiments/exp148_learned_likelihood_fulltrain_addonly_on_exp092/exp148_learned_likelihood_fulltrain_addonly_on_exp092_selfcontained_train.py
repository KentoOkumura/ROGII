# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
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
# # exp148_learned_likelihood_fulltrain_addonly_on_exp092 train
#
# Self-contained train notebook source for exp148. Local experiment helper code is inlined below so Kaggle execution does not import sibling `.py` files.

# %% [markdown]
# ## Contents
#
# 1. Runtime and configuration helpers
# 2. Feature engineering helpers
# 3. Candidate observation features
# 4. Learned likelihood feature engineering
# 5. Model and inference utilities
# 6. Setup and configuration
# 7. Input and full-train coverage contract
# 8. Train full-row control and learned-feature variants
# 9. Metrics and generated artifacts

# %% [markdown]
# ## 1. Runtime and configuration helpers

# %%
from __future__ import annotations

# Inlined from settings.py
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

EXPERIMENT_NAME = "exp148_learned_likelihood_fulltrain_addonly_on_exp092"
PACKAGE_DIR = Path.cwd()
TODO_VALUES = {"", "TODO", "TBD", "FIXME", None}
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")


def find_project_root(start: Path = PACKAGE_DIR) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "project.yml").exists() and (candidate / "AGENTS.md").exists():
            return candidate
    for candidate in (start, *start.parents):
        if (candidate / "project.yml").exists():
            return candidate
    return start


ROOT = find_project_root()


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as fp:
        value = yaml.safe_load(fp) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def get_nested(config: dict[str, Any], dotted_key: str) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        base_value = merged.get(key)
        if isinstance(base_value, dict) and isinstance(value, dict):
            merged[key] = deep_merge(base_value, value)
        else:
            merged[key] = value
    return merged


def first_submission_target(project_config: dict[str, Any]) -> Any:
    target_columns = get_nested(project_config, "submission.target_columns")
    if isinstance(target_columns, list) and target_columns:
        return target_columns[0]
    return None


def load_project_config() -> dict[str, Any]:
    return read_yaml(ROOT / "project.yml")


def is_todo_value(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip() in TODO_VALUES
    try:
        return value in TODO_VALUES
    except TypeError:
        return False


def is_kaggle_runtime() -> bool:
    return KAGGLE_INPUT_ROOT.exists() and KAGGLE_WORKING_ROOT.exists()


def allow_local_notebook_execution() -> bool:
    return os.environ.get("EXPERIMENT_ALLOW_LOCAL", "0") == "1"


def kaggle_competition_input_dir(project_config: dict[str, Any]) -> Path | None:
    slug = get_nested(project_config, "competition.slug")
    input_root = KAGGLE_INPUT_ROOT
    if not input_root.exists():
        return None
    if not is_todo_value(slug):
        candidate = input_root / str(slug)
        if candidate.exists():
            return candidate
    for candidate in sorted(input_root.iterdir()):
        if not candidate.is_dir():
            continue
        if (candidate / "train").is_dir() and (candidate / "test").is_dir():
            return candidate
        if (candidate / "sample_submission.csv").exists():
            return candidate
    for candidate in sorted(input_root.rglob("sample_submission.csv")):
        parent = candidate.parent
        if (parent / "train").is_dir() or (parent / "test").is_dir():
            return parent
    return None


def project_experiment_defaults(project_config: dict[str, Any]) -> dict[str, Any]:
    if not project_config:
        return {}

    data_dir = get_nested(project_config, "paths.data_dir") or "data"
    raw_dir = get_nested(project_config, "data.raw_dir") or f"{data_dir}/raw"
    processed_dir = get_nested(project_config, "data.processed_dir") or f"{data_dir}/processed"
    defaults: dict[str, Any] = {
        "validation": {
            "strategy": get_nested(project_config, "defaults.primary_validation"),
            "n_folds": get_nested(project_config, "defaults.n_folds"),
            "seed": get_nested(project_config, "defaults.seed"),
            "metric": get_nested(project_config, "defaults.metric"),
            "group_column": get_nested(project_config, "data.group_column"),
            "score_rows": get_nested(project_config, "data.score_rows"),
        },
        "data": {
            "raw_dir": raw_dir,
            "train_dir": get_nested(project_config, "data.train_dir") or f"{raw_dir}/train",
            "test_dir": get_nested(project_config, "data.test_dir") or f"{raw_dir}/test",
            "processed_dir": processed_dir,
            "sample_submission": get_nested(project_config, "submission.sample_file"),
            "target_column": get_nested(project_config, "data.target_column"),
            "id_column": get_nested(project_config, "submission.id_column"),
            "submission_target_column": first_submission_target(project_config),
        },
        "project": {
            "competition": project_config.get("competition", {}),
            "submission": project_config.get("submission", {}),
        },
    }

    runtime = project_config.get("runtime")
    if isinstance(runtime, dict):
        defaults["runtime"] = {"kaggle": runtime.get("kaggle", {})}

    return defaults


def load_config() -> dict[str, Any]:
    project_config = load_project_config()
    config_path = PACKAGE_DIR / "config.yaml"
    experiment_config = read_yaml(config_path)
    return deep_merge(project_experiment_defaults(project_config), experiment_config)


@dataclass(frozen=True)
class ExperimentPaths:
    experiment_name: str = EXPERIMENT_NAME

    @property
    def config(self) -> dict[str, Any]:
        return load_config()

    @property
    def root(self) -> Path:
        return ROOT

    @property
    def output_root(self) -> Path:
        if is_kaggle_runtime():
            return KAGGLE_WORKING_ROOT
        return self.root

    @property
    def experiment_dir(self) -> Path:
        if is_kaggle_runtime():
            return KAGGLE_WORKING_ROOT
        experiments_dir = self.resolve_project_path("paths.experiments_dir", "experiments")
        candidate = experiments_dir / self.experiment_name
        if experiments_dir.exists() or candidate.exists():
            return candidate
        return PACKAGE_DIR

    @property
    def data_dir(self) -> Path:
        return self.resolve_project_path("paths.data_dir", "data")

    @property
    def raw_data_dir(self) -> Path:
        local_path = self.resolve_config_path("data.raw_dir", self.data_dir / "raw")
        return self.kaggle_path_or_local(local_path)

    @property
    def train_data_dir(self) -> Path:
        local_path = self.resolve_config_path("data.train_dir", self.raw_data_dir / "train")
        return self.kaggle_path_or_local(local_path, "train")

    @property
    def test_data_dir(self) -> Path:
        local_path = self.resolve_config_path("data.test_dir", self.raw_data_dir / "test")
        return self.kaggle_path_or_local(local_path, "test")

    @property
    def sample_submission_path(self) -> Path:
        local_path = self.resolve_config_path(
            "data.sample_submission",
            self.raw_data_dir / "sample_submission.csv",
        )
        return self.kaggle_path_or_local(local_path, "sample_submission.csv")

    @property
    def processed_data_dir(self) -> Path:
        if is_kaggle_runtime():
            return self.output_root / "data" / "processed"
        return self.resolve_config_path("data.processed_dir", self.data_dir / "processed")

    @property
    def artifacts_dir(self) -> Path:
        return self.experiment_dir / "artifacts"

    @property
    def features_dir(self) -> Path:
        return self.experiment_dir / "features"

    @property
    def metrics_path(self) -> Path:
        return self.experiment_dir / "metrics.json"

    @property
    def submission_path(self) -> Path:
        output_file = (
            get_nested(load_project_config(), "submission.output_file") or "submission.csv"
        )
        path = Path(str(output_file))
        if path.is_absolute():
            return path
        return self.output_root / path

    def resolve_path(self, value: Any, default: str | Path) -> Path:
        path_value = default if value in TODO_VALUES else value
        path = path_value if isinstance(path_value, Path) else Path(str(path_value))
        if path.is_absolute():
            return path
        return self.root / path

    def resolve_config_path(self, dotted_key: str, default: str | Path) -> Path:
        return self.resolve_path(get_nested(self.config, dotted_key), default)

    def resolve_project_path(self, dotted_key: str, default: str | Path) -> Path:
        return self.resolve_path(get_nested(load_project_config(), dotted_key), default)

    def kaggle_path_or_local(self, local_path: Path, relative: str | None = None) -> Path:
        if is_kaggle_runtime():
            kaggle_input = kaggle_competition_input_dir(load_project_config())
            if kaggle_input is None:
                raise FileNotFoundError(
                    "Kaggle runtime detected, but no competition input directory was found "
                    f"under {KAGGLE_INPUT_ROOT}."
                )
            if relative is None:
                return kaggle_input
            candidate = kaggle_input / relative
            if not candidate.exists():
                raise FileNotFoundError(f"Kaggle input path not found: {candidate}")
            return candidate
        if local_path.exists():
            return local_path
        kaggle_input = kaggle_competition_input_dir(load_project_config())
        if kaggle_input is None:
            return local_path
        if relative is None:
            return kaggle_input
        candidate = kaggle_input / relative
        return candidate if candidate.exists() else local_path

    def require_kaggle_runtime(self) -> None:
        if is_kaggle_runtime() or allow_local_notebook_execution():
            return
        raise RuntimeError(
            "Notebook execution is configured for Kaggle. "
            "Use task prepare-kaggle-notebooks and push the generated kernel; "
            "set EXPERIMENT_ALLOW_LOCAL=1 only for an explicit local smoke run."
        )

    def ensure_output_dirs(self) -> None:
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.features_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)


# %% [markdown]
# ## 2. Feature engineering helpers
#
# Public replay feature generation, PF/Beam candidate construction, projection features,
# and base feature schema helpers used by exp148.

# %%
# Inlined from public_notebook_replay_audit.py
# ruff: noqa

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


# %% [markdown]
# ## 3. Candidate observation features
#
# Multi-observation likelihood features used to enrich PF/Beam candidate columns.

# %%
# Inlined from pf_multi_observation_likelihood_probe.py
import argparse
import gzip
import hashlib
import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
# settings helpers are inlined above; no local module import.

EXP072_ARTIFACTS = Path("experiments") / "exp072_exp063_full_replay_feature_cache" / "artifacts"
FULL_REPLAY_TRAIN_FEATURES = (
    "exp063_full_replay_feature_cache_pixiux_likpf_public_replay_train_features.csv.gz"
)
FULL_REPLAY_FEATURE_SCHEMA = "exp063_full_replay_feature_cache_feature_schema.csv"
OUTPUT_PREFIX = "exp099_pf_multi_observation_likelihood_probe"
TRAIN_FEATURE_CACHE_VARIANT = "multiobs_likelihood_probe"
TRAIN_FEATURE_CACHE_FILENAME = (
    f"{OUTPUT_PREFIX}_{TRAIN_FEATURE_CACHE_VARIANT}_train_features.csv.gz"
)
TRAIN_FEATURE_SCHEMA_FILENAME = f"{OUTPUT_PREFIX}_{TRAIN_FEATURE_CACHE_VARIANT}_feature_schema.csv"


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    source_column: str
    transform: str
    role: str
    enabled: bool = True


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [to_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value) if np.isfinite(value) else None
    if pd.isna(value) and not isinstance(value, str):
        return None
    return value


def sha256_path(path: Path, *, decompressed: bool = False) -> str:
    digest = hashlib.sha256()
    opener = gzip.open if decompressed else Path.open
    with opener(path, "rb") as fp:  # type: ignore[arg-type]
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_artifact(
    filename: str,
    explicit_path: str | Path | None = None,
    *,
    local_artifacts: Path = EXP072_ARTIFACTS,
) -> Path:
    candidates: list[Path] = []
    if explicit_path is not None:
        candidates.append(Path(explicit_path))
    candidates.extend(
        [
            local_artifacts / filename,
            Path.cwd() / filename,
            Path.cwd() / "artifacts" / filename,
        ]
    )
    if KAGGLE_INPUT_ROOT.exists():
        candidates.extend(KAGGLE_INPUT_ROOT.glob(f"**/{filename}"))
    for candidate in candidates:
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    checked = "\n".join(str(path) for path in candidates[:80])
    raise FileNotFoundError(f"artifact not found or empty: {filename}. Checked:\n{checked}")


def numeric_array(frame: pd.DataFrame, column: str, *, default: float | None = None) -> np.ndarray:
    if column not in frame.columns:
        if default is None:
            raise ValueError(f"required column is missing: {column}")
        return np.full(len(frame), float(default), dtype=np.float32)
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(np.float32)


def _row_indices_from_ids(ids: pd.Series) -> np.ndarray:
    extracted = ids.astype(str).str.extract(r"_(\d+)$", expand=False)
    values = pd.to_numeric(extracted, errors="coerce").to_numpy()
    if np.isnan(values).any():
        bad = ids[pd.isna(extracted)].head(5).tolist()
        raise ValueError(f"Could not recover row index from ids, examples={bad}")
    return values.astype(np.int32)


def _distance_bucket(values: pd.Series | np.ndarray) -> pd.Categorical:
    return pd.cut(
        pd.to_numeric(values, errors="coerce"),
        bins=[-np.inf, 50.0, 100.0, 250.0, 500.0, 1000.0, np.inf],
        labels=["000_050", "050_100", "100_250", "250_500", "500_1000", "1000_plus"],
        include_lowest=True,
    )


def _tail_rank_bucket(ids: pd.Series) -> pd.Categorical:
    ranks = _row_indices_from_ids(ids)
    return pd.cut(
        ranks,
        bins=[-np.inf, 99, 249, 499, 999, np.inf],
        labels=["000_099", "100_249", "250_499", "500_999", "1000_plus"],
        include_lowest=True,
    )


def _quantile_bucket(values: pd.Series | np.ndarray, prefix: str) -> pd.Categorical:
    series = pd.to_numeric(pd.Series(values), errors="coerce")
    finite = series[np.isfinite(series)]
    if finite.nunique(dropna=True) < 4:
        return pd.Categorical([f"{prefix}_unknown"] * len(series))
    edges = np.unique(np.nanquantile(finite, [0.0, 0.25, 0.50, 0.75, 1.0]))
    if len(edges) < 3:
        return pd.Categorical([f"{prefix}_unknown"] * len(series))
    labels = [f"{prefix}_q{i + 1}" for i in range(len(edges) - 1)]
    return pd.cut(series, bins=edges, labels=labels, include_lowest=True)


def read_feature_cache(
    cache_path: str | Path | None,
    *,
    required_columns: list[str],
    max_rows: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    source = find_artifact(FULL_REPLAY_TRAIN_FEATURES, cache_path)
    header = pd.read_csv(source, nrows=0).columns.tolist()
    missing = [column for column in required_columns if column not in header]
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")
    frame = pd.read_csv(
        source,
        usecols=required_columns,
        nrows=max_rows,
        dtype={"id": str, "well": str},
        low_memory=False,
    )
    frame["id"] = frame["id"].astype(str)
    frame["well"] = frame["well"].astype(str)
    for column in frame.columns:
        if column not in {"id", "well"}:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(np.float32)
    if frame[["id", "well"]].isna().any().any():
        raise ValueError("feature cache contains missing id/well values")
    schema_path: Path | None = None
    try:
        schema_path = find_artifact(FULL_REPLAY_FEATURE_SCHEMA)
    except FileNotFoundError:
        schema_path = None
    metadata = {
        "source": str(source),
        "source_sha256": sha256_path(source),
        "source_decompressed_sha256": (
            sha256_path(source, decompressed=True) if source.suffix == ".gz" else None
        ),
        "schema": str(schema_path) if schema_path else None,
        "schema_sha256": sha256_path(schema_path) if schema_path else None,
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "columns": list(frame.columns),
    }
    return frame, metadata


def build_required_columns(
    candidate_specs: list[CandidateSpec],
    extra_columns: list[str],
) -> list[str]:
    columns = {"id", "well", "target", "last_known_tvt"}
    columns.update(extra_columns)
    for spec in candidate_specs:
        columns.add(spec.source_column)
    return sorted(columns)


def materialize_existing_candidates(
    frame: pd.DataFrame,
    candidate_specs: list[CandidateSpec],
) -> pd.DataFrame:
    out = frame[["id", "well", "target", "last_known_tvt"]].copy()
    last_known = numeric_array(frame, "last_known_tvt")
    out["true_tvt"] = last_known + numeric_array(frame, "target")
    for spec in candidate_specs:
        if not spec.enabled:
            continue
        values = numeric_array(frame, spec.source_column)
        if spec.transform == "absolute":
            out[spec.name] = values
        elif spec.transform == "base_plus_delta":
            out[spec.name] = last_known + values
        else:
            raise ValueError(f"Unsupported candidate transform: {spec.transform}")
    return out


def _nearest_prefix_indices(prefix_tvt: np.ndarray, candidate_tvt: np.ndarray) -> np.ndarray:
    order = np.argsort(prefix_tvt)
    sorted_tvt = prefix_tvt[order]
    positions = np.searchsorted(sorted_tvt, candidate_tvt, side="left")
    left = np.clip(positions - 1, 0, len(sorted_tvt) - 1)
    right = np.clip(positions, 0, len(sorted_tvt) - 1)
    choose_right = np.abs(sorted_tvt[right] - candidate_tvt) < np.abs(
        sorted_tvt[left] - candidate_tvt
    )
    nearest_sorted = np.where(choose_right, right, left)
    return order[nearest_sorted].astype(np.int32)


def _standardize_rows(values: np.ndarray) -> np.ndarray:
    centered = values - values.mean(axis=1, keepdims=True)
    scale = values.std(axis=1, keepdims=True) + 1e-6
    return centered / scale


def _candidate_multi_obs_scores_for_well(
    *,
    full_gr: np.ndarray,
    prefix_tvt: np.ndarray,
    row_idx: np.ndarray,
    candidate_values: np.ndarray,
    observation_offsets: np.ndarray,
    gr_scale: float,
    out_of_range_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_rows, n_candidates = candidate_values.shape
    candidate_values = np.nan_to_num(candidate_values, nan=float(prefix_tvt[-1]))
    nearest_prefix = _nearest_prefix_indices(prefix_tvt, candidate_values.reshape(-1)).reshape(
        n_rows,
        n_candidates,
    )

    eval_vectors = []
    candidate_vectors = []
    for offset in observation_offsets:
        eval_indices = np.clip(row_idx + int(offset), 0, len(full_gr) - 1)
        prefix_indices = np.clip(nearest_prefix + int(offset), 0, len(full_gr) - 1)
        eval_vectors.append(full_gr[eval_indices])
        candidate_vectors.append(full_gr[prefix_indices])
    eval_matrix = np.stack(eval_vectors, axis=1).astype(np.float32)
    candidate_tensor = np.stack(candidate_vectors, axis=2).astype(np.float32)

    diff_mae = np.mean(np.abs(candidate_tensor - eval_matrix[:, None, :]), axis=2)
    eval_norm = _standardize_rows(eval_matrix)
    flat_candidate = candidate_tensor.reshape(n_rows * n_candidates, len(observation_offsets))
    candidate_norm = _standardize_rows(flat_candidate).reshape(
        n_rows,
        n_candidates,
        len(observation_offsets),
    )
    ncc = np.mean(candidate_norm * eval_norm[:, None, :], axis=2)
    ncc_score = np.clip((ncc + 1.0) / 2.0, 0.0, 1.0)

    low = float(np.nanmin(prefix_tvt))
    high = float(np.nanmax(prefix_tvt))
    below = np.maximum(0.0, low - candidate_values)
    above = np.maximum(0.0, candidate_values - high)
    range_penalty = np.exp(-((below + above) / max(out_of_range_scale, 1e-6)))
    mae_score = np.exp(-(diff_mae / max(gr_scale, 1e-6)))
    score = np.clip(mae_score * (0.25 + 0.75 * ncc_score) * range_penalty, 0.0, 1.0)
    return score.astype(np.float32), diff_mae.astype(np.float32), ncc.astype(np.float32)


def build_multi_observation_candidate_frame(
    frame: pd.DataFrame,
    existing_candidates: pd.DataFrame,
    *,
    train_dir: str | Path,
    candidate_names: list[str],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_dir = Path(train_dir)
    observation_offsets = np.asarray(
        [int(value) for value in config.get("observation_offsets", [-24, -12, 0, 12, 24])],
        dtype=np.int32,
    )
    if observation_offsets.size == 0:
        raise ValueError("multi_observation_likelihood.observation_offsets must not be empty")
    gr_rolling_window = int(config.get("gr_rolling_window", 5))
    gr_scale = float(config.get("gr_scale", 18.0))
    out_of_range_scale = float(config.get("out_of_range_scale", 80.0))
    softmax_temperatures = [
        float(value) for value in config.get("softmax_temperatures", [0.15, 0.30])
    ]
    blend_weights = [float(value) for value in config.get("likpf_blend_weights", [0.25, 0.50])]

    base = pd.DataFrame({"id": frame["id"].astype(str), "well": frame["well"].astype(str)})
    base["_row_idx"] = _row_indices_from_ids(base["id"])
    score_frames: list[pd.DataFrame] = []
    well_rows: list[dict[str, Any]] = []
    for well, positions in base.groupby("well", sort=False).groups.items():
        positions_list = list(positions)
        horizontal_path = train_dir / f"{well}__horizontal_well.csv"
        if not horizontal_path.exists():
            raise FileNotFoundError(f"raw train horizontal well file not found: {horizontal_path}")
        horizontal = pd.read_csv(horizontal_path, usecols=["GR", "TVT_input"])
        tvt_input = pd.to_numeric(horizontal["TVT_input"], errors="coerce")
        known_mask = tvt_input.notna().to_numpy()
        if not known_mask.any():
            raise ValueError(f"No finite TVT_input prefix rows for well {well}")
        prefix_len = int(np.flatnonzero(known_mask)[-1] + 1)
        prefix_tvt = (
            tvt_input.iloc[:prefix_len]
            .interpolate(limit_direction="both")
            .ffill()
            .bfill()
            .to_numpy(np.float32)
        )
        if not np.isfinite(prefix_tvt).all():
            raise ValueError(f"Non-finite prefix TVT after interpolation for well {well}")
        gr_series = pd.to_numeric(horizontal["GR"], errors="coerce")
        fallback = float(gr_series.iloc[:prefix_len].mean())
        if not np.isfinite(fallback):
            fallback = float(gr_series.mean()) if np.isfinite(float(gr_series.mean())) else 0.0
        full_gr = (
            gr_series.interpolate(limit_direction="both")
            .fillna(fallback)
            .rolling(gr_rolling_window, center=True, min_periods=1)
            .mean()
            .to_numpy(np.float32)
        )
        row_idx = base.loc[positions_list, "_row_idx"].to_numpy(np.int32)
        if row_idx.min(initial=0) < 0 or row_idx.max(initial=0) >= len(horizontal):
            raise ValueError(f"row index out of range for well {well}")
        candidate_values = existing_candidates.loc[positions_list, candidate_names].to_numpy(
            np.float32
        )
        score, mae, ncc = _candidate_multi_obs_scores_for_well(
            full_gr=full_gr,
            prefix_tvt=prefix_tvt,
            row_idx=row_idx,
            candidate_values=candidate_values,
            observation_offsets=observation_offsets,
            gr_scale=gr_scale,
            out_of_range_scale=out_of_range_scale,
        )
        best_pos = score.argmax(axis=1)
        top1 = candidate_values[np.arange(len(row_idx)), best_pos]
        rows = pd.DataFrame(
            {
                "id": base.loc[positions_list, "id"].to_numpy(),
                "well": str(well),
                "multiobs_top1": top1.astype(np.float32),
                "multiobs_score_max": score.max(axis=1).astype(np.float32),
                "multiobs_score_mean": score.mean(axis=1).astype(np.float32),
                "multiobs_score_gap": (
                    np.sort(score, axis=1)[:, -1] - np.sort(score, axis=1)[:, -2]
                    if score.shape[1] > 1
                    else score[:, 0]
                ).astype(np.float32),
                "multiobs_top1_source_id": best_pos.astype(np.float32),
                "multiobs_top1_mae": mae[np.arange(len(row_idx)), best_pos].astype(np.float32),
                "multiobs_top1_ncc": ncc[np.arange(len(row_idx)), best_pos].astype(np.float32),
            }
        )
        for i, name in enumerate(candidate_names):
            rows[f"multiobs_score_{name}"] = score[:, i].astype(np.float32)
            rows[f"multiobs_mae_{name}"] = mae[:, i].astype(np.float32)
            rows[f"multiobs_ncc_{name}"] = ncc[:, i].astype(np.float32)
        for temp in softmax_temperatures:
            logits = score / max(temp, 1e-6)
            logits = logits - logits.max(axis=1, keepdims=True)
            weights = np.exp(logits)
            weights /= weights.sum(axis=1, keepdims=True) + 1e-9
            key = str(temp).replace(".", "p")
            rows[f"multiobs_softmax_t{key}"] = (candidate_values * weights).sum(axis=1)
        if "likpf_mean" in candidate_names:
            likpf = existing_candidates.loc[positions_list, "likpf_mean"].to_numpy(np.float32)
            for weight in blend_weights:
                key = str(weight).replace(".", "p")
                rows[f"likpf_multiobs_blend_w{key}"] = (
                    (1.0 - weight) * likpf + weight * top1
                ).astype(np.float32)
        score_frames.append(rows)
        well_rows.append(
            {
                "well": str(well),
                "rows": int(len(row_idx)),
                "known_prefix_rows": int(prefix_len),
                "eval_len": int(max(0, len(horizontal) - prefix_len)),
                "gr_missing_rate": float(pd.isna(horizontal["GR"]).mean()),
                "multiobs_score_mean": float(np.mean(score)),
                "multiobs_score_p10": float(np.quantile(score, 0.10)),
                "multiobs_score_p90": float(np.quantile(score, 0.90)),
            }
        )
    out = pd.concat(score_frames, ignore_index=True)
    if out.drop(columns=["id", "well"]).isna().any().any():
        raise ValueError("multi-observation candidate frame contains missing values")
    return out, pd.DataFrame(well_rows)


def _candidate_score(
    frame: pd.DataFrame,
    candidate_name: str,
    *,
    multi_obs_columns: set[str],
) -> np.ndarray:
    n = len(frame)
    if candidate_name in multi_obs_columns:
        return np.clip(numeric_array(frame, "multiobs_score_max", default=0.0), 0.0, 1.0)
    score_col = f"multiobs_score_{candidate_name}"
    if score_col in frame.columns:
        return np.clip(numeric_array(frame, score_col, default=0.0), 0.0, 1.0)
    if candidate_name == "last_anchor_tvt":
        return np.full(n, 0.10, dtype=np.float32)
    return np.full(n, 0.25, dtype=np.float32)


def build_candidate_long_frame(
    base_frame: pd.DataFrame,
    candidate_columns: list[str],
    *,
    multi_obs_columns: set[str],
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    true_tvt = numeric_array(base_frame, "true_tvt")
    for name in candidate_columns:
        if name not in base_frame.columns:
            continue
        pred = numeric_array(base_frame, name)
        score = _candidate_score(base_frame, name, multi_obs_columns=multi_obs_columns)
        family = "multi_observation_likelihood" if name in multi_obs_columns else "existing"
        item = pd.DataFrame(
            {
                "id": base_frame["id"].to_numpy(),
                "well": base_frame["well"].to_numpy(),
                "candidate": name,
                "pred_tvt": pred,
                "target_tvt": true_tvt,
                "abs_error": np.abs(pred - true_tvt).astype(np.float32),
                "rank_score": score,
                "candidate_family": family,
            }
        )
        rows.append(item)
    long_frame = pd.concat(rows, ignore_index=True)
    if not np.isfinite(
        long_frame[["pred_tvt", "target_tvt", "abs_error", "rank_score"]].to_numpy()
    ).all():
        raise ValueError("candidate long frame contains non-finite numeric values")
    return long_frame


def write_train_feature_cache(
    *,
    output_dir: Path,
    source_frame: pd.DataFrame,
    full_frame: pd.DataFrame,
    candidate_columns: list[str],
) -> dict[str, Any]:
    meta_columns = ["id", "well", "target"]
    missing_meta = [column for column in meta_columns if column not in source_frame.columns]
    if missing_meta:
        raise ValueError(
            f"source frame is missing train feature cache meta columns: {missing_meta}"
        )

    feature_columns: list[str] = []
    for column in source_frame.columns:
        if column not in {"id", "well", "target"}:
            feature_columns.append(column)
    for column in candidate_columns:
        if column not in {"id", "well", "target", "true_tvt"}:
            feature_columns.append(column)
    for column in full_frame.columns:
        if (
            column.startswith("multiobs_") or column.startswith("likpf_multiobs_blend_")
        ) and column not in {"id", "well", "target", "true_tvt"}:
            feature_columns.append(column)

    feature_columns = list(dict.fromkeys(feature_columns))
    if not feature_columns:
        raise ValueError("train feature cache feature column list is empty")

    train_frame = source_frame[meta_columns].copy()
    for column in feature_columns:
        if column in full_frame.columns:
            values = full_frame[column]
        elif column in source_frame.columns:
            values = source_frame[column]
        else:
            raise ValueError(f"train feature cache column is missing: {column}")
        train_frame[column] = pd.to_numeric(values, errors="coerce").astype(np.float32)

    numeric_values = train_frame[["target", *feature_columns]].to_numpy(np.float32)
    if not np.isfinite(numeric_values).all():
        bad_columns = [
            column
            for column in ["target", *feature_columns]
            if not np.isfinite(train_frame[column].to_numpy(np.float32)).all()
        ]
        raise ValueError(f"train feature cache contains non-finite values: {bad_columns[:20]}")

    train_path = output_dir / TRAIN_FEATURE_CACHE_FILENAME
    schema_path = output_dir / TRAIN_FEATURE_SCHEMA_FILENAME
    train_frame.to_csv(train_path, index=False, compression="gzip")
    pd.DataFrame(
        {
            "variant": TRAIN_FEATURE_CACHE_VARIANT,
            "feature_index": np.arange(len(feature_columns), dtype=np.int32),
            "feature": feature_columns,
        }
    ).to_csv(schema_path, index=False)

    return {
        "variant": TRAIN_FEATURE_CACHE_VARIANT,
        "rows": int(len(train_frame)),
        "wells": int(train_frame["well"].nunique()),
        "feature_count": int(len(feature_columns)),
        "feature_columns": feature_columns,
        "outputs": {
            "train_features": train_path.name,
            "train_feature_schema": schema_path.name,
        },
        "sha256": {
            "train_features": sha256_path(train_path),
            "train_features_decompressed": sha256_path(train_path, decompressed=True),
            "train_feature_schema": sha256_path(schema_path),
        },
    }


def summarize_candidate_metrics(long_frame: pd.DataFrame, thresholds: list[float]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate, group in long_frame.groupby("candidate", sort=False):
        error = group["pred_tvt"].to_numpy(np.float32) - group["target_tvt"].to_numpy(np.float32)
        abs_error = np.abs(error)
        row: dict[str, Any] = {
            "candidate": str(candidate),
            "candidate_family": str(group["candidate_family"].iloc[0]),
            "rows": int(len(group)),
            "rmse_tvt": float(np.sqrt(np.mean(np.square(error)))),
            "mae_tvt": float(np.mean(abs_error)),
            "bias_tvt": float(np.mean(error)),
            "abs_error_p50": float(np.quantile(abs_error, 0.50)),
            "abs_error_p90": float(np.quantile(abs_error, 0.90)),
            "abs_error_p95": float(np.quantile(abs_error, 0.95)),
            "rank_score_mean": float(group["rank_score"].mean()),
        }
        for threshold in thresholds:
            row[f"within_{threshold:g}ft"] = float(np.mean(abs_error <= float(threshold)))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("rmse_tvt").reset_index(drop=True)


def summarize_rank_metrics(
    long_frame: pd.DataFrame,
    thresholds: list[float],
    topk_values: list[int],
    candidate_sets: dict[str, list[str]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate_set, candidates in candidate_sets.items():
        subset_frame = long_frame[long_frame["candidate"].isin(candidates)].copy()
        candidate_count = int(subset_frame["candidate"].nunique())
        if candidate_count == 0:
            continue
        for rank_family, sorted_frame in {
            "oracle_best_error": subset_frame.sort_values(["id", "abs_error"]),
            "candidate_rank_score": subset_frame.sort_values(
                ["id", "rank_score"],
                ascending=[True, False],
            ),
        }.items():
            for topk in topk_values:
                k = min(int(topk), candidate_count)
                subset = sorted_frame.groupby("id", sort=False).head(k)
                best = subset.sort_values(["id", "abs_error"]).groupby("id", sort=False).head(1)
                error = best["pred_tvt"].to_numpy(np.float32) - best["target_tvt"].to_numpy(
                    np.float32
                )
                abs_error = np.abs(error)
                row: dict[str, Any] = {
                    "candidate_set": candidate_set,
                    "rank_family": rank_family,
                    "topk": int(k),
                    "candidate_count": int(candidate_count),
                    "rows": int(len(best)),
                    "rmse_tvt": float(np.sqrt(np.mean(np.square(error)))),
                    "mae_tvt": float(np.mean(abs_error)),
                    "selected_multiobs_rate": float(
                        best["candidate_family"].eq("multi_observation_likelihood").mean()
                    ),
                    "selected_candidate_top": str(best["candidate"].mode().iloc[0]),
                }
                for threshold in thresholds:
                    row[f"within_{threshold:g}ft"] = float(np.mean(abs_error <= float(threshold)))
                rows.append(row)
    return pd.DataFrame(rows)


def build_row_context(source_frame: pd.DataFrame, full_frame: pd.DataFrame) -> pd.DataFrame:
    context = source_frame[["id", "well"]].copy()
    context["distance_bucket"] = _distance_bucket(source_frame.get("md_since", np.nan))
    context["tail_rank_bucket"] = _tail_rank_bucket(source_frame["id"])
    for source_column, bucket_name in [
        ("eval_len", "eval_len_bucket"),
        ("pf_ancc_std", "pf_seed_std_bucket"),
        ("likpf_mean_d", "likpf_delta_bucket"),
    ]:
        if source_column in source_frame.columns:
            context[bucket_name] = _quantile_bucket(source_frame[source_column], bucket_name)
        else:
            context[bucket_name] = pd.Categorical([f"{bucket_name}_unknown"] * len(context))
    context["multiobs_score_bucket"] = _quantile_bucket(
        full_frame["multiobs_score_max"],
        "multiobs_score",
    )
    return context


def summarize_bucket_metrics(
    long_frame: pd.DataFrame,
    context_frame: pd.DataFrame,
    thresholds: list[float],
) -> pd.DataFrame:
    frame = long_frame.merge(context_frame, on=["id", "well"], how="left", validate="many_to_one")
    bucket_families = [
        "distance_bucket",
        "tail_rank_bucket",
        "eval_len_bucket",
        "pf_seed_std_bucket",
        "likpf_delta_bucket",
        "multiobs_score_bucket",
    ]
    rows: list[dict[str, Any]] = []
    for bucket_family in bucket_families:
        for (candidate, bucket), group in frame.groupby(
            ["candidate", bucket_family],
            observed=True,
        ):
            error = group["pred_tvt"].to_numpy(np.float32) - group["target_tvt"].to_numpy(
                np.float32
            )
            abs_error = np.abs(error)
            row: dict[str, Any] = {
                "candidate": str(candidate),
                "bucket_family": bucket_family,
                "bucket": str(bucket),
                "rows": int(len(group)),
                "rmse_tvt": float(np.sqrt(np.mean(np.square(error)))),
                "mae_tvt": float(np.mean(abs_error)),
            }
            for threshold in thresholds:
                row[f"miss_gt_{threshold:g}ft"] = float(np.mean(abs_error > float(threshold)))
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_by_well(long_frame: pd.DataFrame, thresholds: list[float]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (candidate, well), group in long_frame.groupby(["candidate", "well"], sort=False):
        error = group["pred_tvt"].to_numpy(np.float32) - group["target_tvt"].to_numpy(np.float32)
        abs_error = np.abs(error)
        row: dict[str, Any] = {
            "candidate": str(candidate),
            "well": str(well),
            "rows": int(len(group)),
            "rmse_tvt": float(np.sqrt(np.mean(np.square(error)))),
            "mae_tvt": float(np.mean(abs_error)),
        }
        for threshold in thresholds:
            row[f"within_{threshold:g}ft"] = float(np.mean(abs_error <= float(threshold)))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["candidate", "rmse_tvt"], ascending=[True, False])


def candidate_sets_from_config(config: dict[str, Any]) -> dict[str, list[str]]:
    configured = get_nested(config, "audit.candidate_sets") or []
    out: dict[str, list[str]] = {}
    for item in configured:
        out[str(item["name"])] = [str(candidate) for candidate in item.get("candidates", [])]
    if not out:
        raise ValueError("audit.candidate_sets must configure at least one candidate set")
    return out


def summarize_probe_decision(rank_metrics: pd.DataFrame, *, primary_threshold_ft: float) -> dict:
    within_col = f"within_{primary_threshold_ft:g}ft"
    oracle = rank_metrics[
        (rank_metrics["rank_family"] == "oracle_best_error")
        & (rank_metrics["topk"] == rank_metrics["candidate_count"])
    ]
    rank_top1 = rank_metrics[
        (rank_metrics["rank_family"] == "candidate_rank_score") & (rank_metrics["topk"] == 1)
    ]
    by_set = {str(row["candidate_set"]): row for _, row in oracle.iterrows()}
    baseline = by_set.get("baseline_primary")
    expanded = by_set.get("baseline_plus_multiobs")
    rmse_gain = None
    coverage_gain = None
    if baseline is not None and expanded is not None:
        rmse_gain = float(baseline["rmse_tvt"] - expanded["rmse_tvt"])
        coverage_gain = float(expanded[within_col] - baseline[within_col])
    return {
        "primary_threshold_ft": float(primary_threshold_ft),
        "baseline_primary_oracle": (
            to_jsonable(baseline.to_dict()) if baseline is not None else None
        ),
        "baseline_plus_multiobs_oracle": (
            to_jsonable(expanded.to_dict()) if expanded is not None else None
        ),
        "oracle_rmse_gain_from_multiobs": rmse_gain,
        "oracle_coverage_gain_from_multiobs": coverage_gain,
        "candidate_rank_score_top1": {
            str(row["candidate_set"]): to_jsonable(row.to_dict()) for _, row in rank_top1.iterrows()
        },
        "recommendation": (
            "multi_observation_likelihood_supported_for_ranker_features"
            if rmse_gain is not None and rmse_gain > 0.0
            else "do_not_add_multi_observation_candidates_without_better_scorer"
        ),
    }


def run_pf_multi_observation_likelihood_probe(
    *,
    output_dir: str | Path,
    train_dir: str | Path,
    cache_path: str | Path | None,
    candidate_specs: list[CandidateSpec],
    extra_source_columns: list[str],
    multi_obs_config: dict[str, Any],
    candidate_sets: dict[str, list[str]],
    thresholds: list[float],
    topk_values: list[int],
    max_rows: int | None = None,
    save_candidate_long: bool = True,
) -> dict[str, Any]:
    t0 = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    required_columns = build_required_columns(candidate_specs, extra_source_columns)
    source_frame, source_meta = read_feature_cache(
        cache_path,
        required_columns=required_columns,
        max_rows=max_rows,
    )
    existing = materialize_existing_candidates(source_frame, candidate_specs)
    likelihood_candidate_names = [
        str(name)
        for name in multi_obs_config.get(
            "score_candidates",
            ["pf_ancc", "beam_mean", "likpf_mean", "sc_ens", "hyb"],
        )
    ]
    missing = [name for name in likelihood_candidate_names if name not in existing.columns]
    if missing:
        raise ValueError(f"multi-observation score candidates missing from frame: {missing}")
    multi_obs_frame, multi_obs_well_summary = build_multi_observation_candidate_frame(
        source_frame,
        existing,
        train_dir=train_dir,
        candidate_names=likelihood_candidate_names,
        config=multi_obs_config,
    )
    full_frame = existing.merge(
        multi_obs_frame,
        on=["id", "well"],
        how="left",
        validate="one_to_one",
    )
    if full_frame.isna().any().any():
        raise ValueError("candidate merge produced missing values")

    existing_candidate_columns = [spec.name for spec in candidate_specs if spec.enabled]
    multi_obs_candidate_columns = [
        column
        for column in full_frame.columns
        if column.startswith("multiobs_softmax_")
        or column.startswith("likpf_multiobs_blend_")
        or column == "multiobs_top1"
    ]
    candidate_columns = existing_candidate_columns + multi_obs_candidate_columns
    long_frame = build_candidate_long_frame(
        full_frame,
        candidate_columns,
        multi_obs_columns=set(multi_obs_candidate_columns),
    )
    context_frame = build_row_context(source_frame, full_frame)
    train_feature_cache = write_train_feature_cache(
        output_dir=output_dir,
        source_frame=source_frame,
        full_frame=full_frame,
        candidate_columns=candidate_columns,
    )

    candidate_metrics = summarize_candidate_metrics(long_frame, thresholds)
    rank_metrics = summarize_rank_metrics(long_frame, thresholds, topk_values, candidate_sets)
    bucket_metrics = summarize_bucket_metrics(long_frame, context_frame, thresholds)
    by_well = summarize_by_well(long_frame, thresholds)
    decision = summarize_probe_decision(
        rank_metrics,
        primary_threshold_ft=float(multi_obs_config.get("primary_threshold_ft", 10.0)),
    )

    candidate_metrics.to_csv(output_dir / f"{OUTPUT_PREFIX}_candidate_metrics.csv", index=False)
    rank_metrics.to_csv(output_dir / f"{OUTPUT_PREFIX}_rank_metrics.csv", index=False)
    bucket_metrics.to_csv(output_dir / f"{OUTPUT_PREFIX}_bucket_metrics.csv", index=False)
    by_well.to_csv(output_dir / f"{OUTPUT_PREFIX}_by_well.csv", index=False)
    multi_obs_well_summary.to_csv(
        output_dir / f"{OUTPUT_PREFIX}_multiobs_well_summary.csv",
        index=False,
    )
    context_frame.to_csv(output_dir / f"{OUTPUT_PREFIX}_row_context.csv.gz", index=False)
    source_schema = pd.DataFrame(
        [{"column": column, "role": "source"} for column in source_frame.columns]
        + [{"column": column, "role": "candidate"} for column in candidate_columns]
        + [
            {"column": column, "role": "multi_observation_likelihood"}
            for column in multi_obs_frame.columns
            if column not in {"id", "well"}
        ]
    )
    source_schema.to_csv(output_dir / f"{OUTPUT_PREFIX}_feature_schema.csv", index=False)
    if save_candidate_long:
        long_frame.to_csv(
            output_dir / f"{OUTPUT_PREFIX}_candidate_long.csv.gz",
            index=False,
            compression="gzip",
        )

    summary = {
        "experiment": OUTPUT_PREFIX,
        "status": "completed_train_side_audit" if max_rows is None else "debug_completed",
        "created_at": datetime.now(UTC).isoformat(),
        "runtime_seconds": float(time.time() - t0),
        "source": source_meta,
        "multi_observation_likelihood": {
            "config": multi_obs_config,
            "score_candidates": likelihood_candidate_names,
            "generated_candidates": multi_obs_candidate_columns,
            "well_summary_rows": int(len(multi_obs_well_summary)),
        },
        "train_feature_cache": train_feature_cache,
        "candidate_sets": candidate_sets,
        "thresholds_ft": thresholds,
        "topk_values": topk_values,
        "best_candidate_by_rmse": to_jsonable(candidate_metrics.iloc[0].to_dict()),
        "probe_decision": to_jsonable(decision),
        "outputs": {
            "candidate_metrics": f"{OUTPUT_PREFIX}_candidate_metrics.csv",
            "rank_metrics": f"{OUTPUT_PREFIX}_rank_metrics.csv",
            "bucket_metrics": f"{OUTPUT_PREFIX}_bucket_metrics.csv",
            "by_well": f"{OUTPUT_PREFIX}_by_well.csv",
            "multiobs_well_summary": f"{OUTPUT_PREFIX}_multiobs_well_summary.csv",
            "train_features": TRAIN_FEATURE_CACHE_FILENAME,
            "train_feature_schema": TRAIN_FEATURE_SCHEMA_FILENAME,
            "candidate_long": f"{OUTPUT_PREFIX}_candidate_long.csv.gz"
            if save_candidate_long
            else None,
            "row_context": f"{OUTPUT_PREFIX}_row_context.csv.gz",
            "feature_schema": f"{OUTPUT_PREFIX}_feature_schema.csv",
            "summary": f"{OUTPUT_PREFIX}_summary.json",
        },
    }
    with (output_dir / f"{OUTPUT_PREFIX}_summary.json").open("w") as fp:
        json.dump(to_jsonable(summary), fp, indent=2, sort_keys=True)
    return summary


def candidate_specs_from_config(config: dict[str, Any]) -> list[CandidateSpec]:
    specs: list[CandidateSpec] = []
    for item in get_nested(config, "audit.candidates") or []:
        specs.append(
            CandidateSpec(
                name=str(item["name"]),
                source_column=str(item.get("source_column") or item["name"]),
                transform=str(item.get("transform", "absolute")),
                role=str(item.get("role", item.get("name", "candidate"))),
                enabled=bool(item.get("enabled", True)),
            )
        )
    if not specs:
        raise ValueError("audit.candidates must configure at least one candidate")
    return specs


def run_from_config(
    config: dict[str, Any] | None = None,
    *,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    config = config or load_config()
    paths = ExperimentPaths()
    paths.ensure_output_dirs()
    return run_pf_multi_observation_likelihood_probe(
        output_dir=output_dir or paths.artifacts_dir,
        train_dir=paths.train_data_dir,
        cache_path=get_nested(config, "data.exp072_train_feature_cache_local"),
        candidate_specs=candidate_specs_from_config(config),
        extra_source_columns=[
            str(col) for col in get_nested(config, "audit.extra_source_columns") or []
        ],
        multi_obs_config=get_nested(config, "model.multi_observation_likelihood") or {},
        candidate_sets=candidate_sets_from_config(config),
        thresholds=[
            float(value) for value in get_nested(config, "audit.thresholds_ft") or [1, 2, 5, 10]
        ],
        topk_values=[
            int(value) for value in get_nested(config, "audit.topk_values") or [1, 2, 3, 5, 10]
        ],
        max_rows=get_nested(config, "audit.max_rows"),
        save_candidate_long=bool(get_nested(config, "audit.save_candidate_long") is not False),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()
    config = load_config()
    if args.max_rows is not None:
        config.setdefault("audit", {})["max_rows"] = args.max_rows
    summary = run_from_config(config, output_dir=args.output_dir)
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))


# Notebook source: keep CLI entry point disabled when helpers are inlined.
# if __name__ == "__main__":
#     main()


# %% [markdown]
# ## 4. Learned likelihood feature engineering
#
# Raw-test and train-frame learned likelihood feature builders ported into this notebook.

# %%
# Inlined from learned_likelihood_rawtest_feature_generator_parity.py
import argparse
import gzip
import hashlib
import json
import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
# settings helpers are inlined above; no local module import.

OUTPUT_PREFIX = "exp145_learned_likelihood_rawtest_feature_generator_parity"
EXP111_PREFIX = "exp111_learned_pf_observation_likelihood_probe"
EXP112_PREFIX = "exp112_learned_pf_likelihood_weight_or_feature_followup"
DEFAULT_EXP099_TRAIN_FEATURES = (
    "exp099_pf_multi_observation_likelihood_probe_multiobs_likelihood_probe_train_features.csv.gz"
)
DEFAULT_EXP099_FEATURE_SCHEMA = (
    "exp099_pf_multi_observation_likelihood_probe_multiobs_likelihood_probe_feature_schema.csv"
)
DEFAULT_EXP111_SCHEMA = f"{EXP111_PREFIX}_feature_schema.csv"
DEFAULT_EXP111_MANIFEST = f"{EXP111_PREFIX}_model_manifest.json"
DEFAULT_EXP112_SCHEMA = f"{EXP112_PREFIX}_feature_schema.csv"


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    column: str


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [to_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value) if np.isfinite(value) else None
    if pd.isna(value) and not isinstance(value, str):
        return None
    return value


def sha256_path(path: Path, *, decompressed: bool = False) -> str:
    digest = hashlib.sha256()
    opener = gzip.open if decompressed else Path.open
    with opener(path, "rb") as fp:  # type: ignore[arg-type]
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_artifact(
    filename: str,
    explicit_path: str | Path | None = None,
    *,
    local_dirs: Iterable[str | Path] = (),
) -> Path:
    candidates: list[Path] = []
    if explicit_path is not None:
        explicit = Path(explicit_path)
        candidates.append(explicit if explicit.name == filename else explicit / filename)
    candidates.extend(Path(path) / filename for path in local_dirs)
    candidates.extend([Path.cwd() / filename, Path.cwd() / "artifacts" / filename])
    if KAGGLE_INPUT_ROOT.exists():
        candidates.extend(KAGGLE_INPUT_ROOT.glob(f"**/{filename}"))
    for candidate in candidates:
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    checked = "\n".join(str(path) for path in candidates[:120])
    raise FileNotFoundError(f"artifact not found or empty: {filename}. Checked:\n{checked}")


def candidate_specs_from_config(config: dict[str, Any]) -> list[CandidateSpec]:
    values = get_nested(config, "generator.candidates") or []
    specs: list[CandidateSpec] = []
    for item in values:
        if not isinstance(item, dict):
            raise ValueError("generator.candidates entries must be mappings")
        specs.append(
            CandidateSpec(name=str(item["name"]), column=str(item.get("column", item["name"])))
        )
    if not specs:
        raise ValueError("generator.candidates must not be empty")
    if "likpf_mean" not in {spec.name for spec in specs}:
        raise ValueError("generator.candidates must include likpf_mean")
    return specs


def load_feature_schema(path: Path) -> list[str]:
    schema = pd.read_csv(path)
    if "feature" not in schema.columns:
        raise ValueError(f"{path} must contain a feature column")
    return [str(value) for value in schema["feature"].tolist()]


def source_required_columns(config: dict[str, Any], candidates: list[CandidateSpec]) -> list[str]:
    required = {"id", "well", "last_known_tvt"}
    required.update(spec.column for spec in candidates)
    for key in ["generator.row_context_columns", "generator.multiobs_global_columns"]:
        required.update(str(value) for value in get_nested(config, key) or [])
    for spec in candidates:
        for suffix in ["score", "mae", "ncc"]:
            required.add(f"multiobs_{suffix}_{spec.name}")
    return sorted(required)


def validate_source_header(source: Path, required_columns: list[str]) -> list[str]:
    header = pd.read_csv(source, nrows=0).columns.tolist()
    missing = [column for column in required_columns if column not in header]
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")
    return header


def numeric_source_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["id"] = frame["id"].astype(str)
    frame["well"] = frame["well"].astype(str)
    for column in frame.columns:
        if column not in {"id", "well"}:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(np.float32)
    return frame


def add_target_free_row_features(
    frame: pd.DataFrame,
    candidates: list[CandidateSpec],
    *,
    include_candidate_values: bool,
) -> tuple[pd.DataFrame, list[str], np.ndarray]:
    out = numeric_source_frame(frame)
    candidate_values = np.column_stack(
        [
            pd.to_numeric(out[spec.column], errors="coerce").to_numpy(np.float32)
            for spec in candidates
        ]
    )
    if not np.isfinite(candidate_values).all():
        bad = np.argwhere(~np.isfinite(candidate_values))[:5].tolist()
        raise ValueError(f"candidate values contain non-finite values, examples={bad}")

    value_cols = [spec.column for spec in candidates]
    out["candidate_mean"] = out[value_cols].mean(axis=1).astype(np.float32)
    out["candidate_std"] = out[value_cols].std(axis=1).astype(np.float32)
    out["candidate_range"] = (out[value_cols].max(axis=1) - out[value_cols].min(axis=1)).astype(
        np.float32
    )

    engineered = ["candidate_mean", "candidate_std", "candidate_range"]
    for spec in candidates:
        delta_col = f"{spec.name}_minus_last"
        out[delta_col] = out[spec.column].astype(np.float32) - out["last_known_tvt"].astype(
            np.float32
        )
        engineered.append(delta_col)
        if include_candidate_values:
            engineered.append(spec.column)

    for i, left in enumerate(candidates):
        for right in candidates[i + 1 :]:
            col = f"{left.name}_vs_{right.name}_abs"
            out[col] = np.abs(
                out[left.column].astype(np.float32) - out[right.column].astype(np.float32)
            )
            engineered.append(col)
    return out, engineered, candidate_values


def rank_desc(values: np.ndarray) -> np.ndarray:
    order = np.argsort(-values, axis=1)
    ranks = np.empty_like(order, dtype=np.int16)
    rows = np.arange(values.shape[0])[:, None]
    ranks[rows, order] = np.arange(values.shape[1], dtype=np.int16)
    return ranks


def rank_asc(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, axis=1)
    ranks = np.empty_like(order, dtype=np.int16)
    rows = np.arange(values.shape[0])[:, None]
    ranks[rows, order] = np.arange(values.shape[1], dtype=np.int16)
    return ranks


def build_candidate_long_features(
    frame: pd.DataFrame,
    candidates: list[CandidateSpec],
    *,
    row_feature_columns: list[str],
    candidate_values: np.ndarray,
) -> pd.DataFrame:
    candidate_names = [spec.name for spec in candidates]
    row_mean = candidate_values.mean(axis=1).astype(np.float32)
    row_std = candidate_values.std(axis=1).astype(np.float32)
    row_std_safe = np.where(row_std > 1e-6, row_std, 1.0).astype(np.float32)
    last_known = frame["last_known_tvt"].to_numpy(np.float32)

    score_cols = [f"multiobs_score_{spec.name}" for spec in candidates]
    mae_cols = [f"multiobs_mae_{spec.name}" for spec in candidates]
    ncc_cols = [f"multiobs_ncc_{spec.name}" for spec in candidates]
    score_matrix = (
        frame[score_cols].replace([np.inf, -np.inf], np.nan).fillna(-1e9).to_numpy(np.float32)
    )
    mae_matrix = frame[mae_cols].replace([np.inf, -np.inf], np.nan).fillna(1e9).to_numpy(np.float32)
    ncc_matrix = frame[ncc_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(np.float32)
    score_max = score_matrix.max(axis=1).astype(np.float32)
    score_mean = score_matrix.mean(axis=1).astype(np.float32)
    mae_min = mae_matrix.min(axis=1).astype(np.float32)
    ncc_max = ncc_matrix.max(axis=1).astype(np.float32)
    score_rank = rank_desc(score_matrix).astype(np.float32)
    mae_rank = rank_asc(mae_matrix).astype(np.float32)
    ncc_rank = rank_desc(ncc_matrix).astype(np.float32)

    chunks: list[pd.DataFrame] = []
    for cand_idx, _spec in enumerate(candidates):
        part = frame[["id", "well", *row_feature_columns]].copy()
        cand = candidate_values[:, cand_idx].astype(np.float32)
        part["candidate_index"] = np.int16(cand_idx)
        part["candidate_name"] = candidate_names[cand_idx]
        part["candidate_tvt"] = cand
        part["candidate_minus_last"] = (cand - last_known).astype(np.float32)
        part["candidate_abs_minus_likpf"] = np.abs(
            cand - frame["likpf_mean"].to_numpy(np.float32)
        ).astype(np.float32)
        part["candidate_abs_minus_row_mean"] = np.abs(cand - row_mean).astype(np.float32)
        part["candidate_z_within_row"] = ((cand - row_mean) / row_std_safe).astype(np.float32)
        part["candidate_multiobs_score"] = score_matrix[:, cand_idx]
        part["candidate_multiobs_mae"] = mae_matrix[:, cand_idx]
        part["candidate_multiobs_ncc"] = ncc_matrix[:, cand_idx]
        part["candidate_score_gap_from_best"] = (score_max - score_matrix[:, cand_idx]).astype(
            np.float32
        )
        part["candidate_score_centered"] = (score_matrix[:, cand_idx] - score_mean).astype(
            np.float32
        )
        part["candidate_mae_gap_from_best"] = (mae_matrix[:, cand_idx] - mae_min).astype(np.float32)
        part["candidate_ncc_gap_from_best"] = (ncc_max - ncc_matrix[:, cand_idx]).astype(np.float32)
        part["candidate_score_rank"] = score_rank[:, cand_idx]
        part["candidate_mae_rank"] = mae_rank[:, cand_idx]
        part["candidate_ncc_rank"] = ncc_rank[:, cand_idx]
        chunks.append(part)
    return pd.concat(chunks, ignore_index=True)


def prepare_model_matrix(long_frame: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    missing = [column for column in feature_columns if column not in long_frame.columns]
    if missing:
        raise ValueError(f"candidate-long frame missing model features: {missing}")
    values = long_frame[feature_columns].replace([np.inf, -np.inf], np.nan).to_numpy(np.float32)
    medians = np.nanmedian(values, axis=0).astype(np.float32)
    medians[~np.isfinite(medians)] = 0.0
    bad = ~np.isfinite(values)
    if bad.any():
        values[bad] = np.take(medians, np.where(bad)[1])
    return values


def exp111_model_feature_columns(row_feature_columns: list[str]) -> list[str]:
    """Reconstruct exp111 long_feature_columns() order for saved numpy-trained boosters."""
    candidate_columns = [
        "candidate_index",
        "candidate_tvt",
        "candidate_minus_last",
        "candidate_abs_minus_likpf",
        "candidate_abs_minus_row_mean",
        "candidate_z_within_row",
        "candidate_multiobs_score",
        "candidate_multiobs_mae",
        "candidate_multiobs_ncc",
        "candidate_score_gap_from_best",
        "candidate_score_centered",
        "candidate_mae_gap_from_best",
        "candidate_ncc_gap_from_best",
        "candidate_score_rank",
        "candidate_mae_rank",
        "candidate_ncc_rank",
    ]
    return [*row_feature_columns, *candidate_columns]


def load_exp111_models(
    *,
    manifest_path: Path,
    model_root: Path | None = None,
) -> tuple[Any, Any, dict[str, Any]]:
    import lightgbm as lgb

    manifest = json.loads(manifest_path.read_text())
    root = model_root or manifest_path.parent
    models = manifest.get("models") or []
    classifier_rows = [item for item in models if item.get("variant") == "within10_classifier"]
    error_rows = [item for item in models if item.get("variant") == "expected_error_regressor"]
    if len(classifier_rows) != 1 or len(error_rows) != 1:
        raise ValueError(f"expected one classifier and one regressor in {manifest_path}")
    classifier_row = classifier_rows[0]
    error_row = error_rows[0]
    classifier_path = root / str(classifier_row["path"])
    error_path = root / str(error_row["path"])
    if not classifier_path.exists() or not error_path.exists():
        raise FileNotFoundError(f"exp111 model files are missing under {root}")
    classifier = lgb.Booster(model_file=str(classifier_path))
    error_model = lgb.Booster(model_file=str(error_path))
    meta = {
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_path(manifest_path),
        "model_root": str(root),
        "classifier": {**classifier_row, "resolved_path": str(classifier_path)},
        "expected_error": {**error_row, "resolved_path": str(error_path)},
        "classifier_sha256_actual": sha256_path(classifier_path),
        "expected_error_sha256_actual": sha256_path(error_path),
    }
    return classifier, error_model, meta


def predict_likelihood_matrices(
    frame: pd.DataFrame,
    candidates: list[CandidateSpec],
    *,
    row_feature_columns: list[str],
    model_feature_columns: list[str],
    classifier: Any,
    error_model: Any,
    candidate_values: np.ndarray,
) -> tuple[dict[str, Any], pd.DataFrame]:
    long_frame = build_candidate_long_features(
        frame,
        candidates,
        row_feature_columns=row_feature_columns,
        candidate_values=candidate_values,
    )
    expected_features = int(classifier.num_feature())
    if len(model_feature_columns) != expected_features:
        raise ValueError(
            f"Reconstructed exp111 model feature count mismatch: "
            f"{len(model_feature_columns)} != booster num_feature {expected_features}"
        )
    x_matrix = prepare_model_matrix(long_frame, model_feature_columns)
    n_rows = len(frame)
    n_candidates = len(candidates)
    probability = classifier.predict(x_matrix).astype(np.float32).reshape(n_candidates, n_rows).T
    pred_error = error_model.predict(x_matrix).astype(np.float32).reshape(n_candidates, n_rows).T
    pred_error = np.maximum(pred_error, 0.0).astype(np.float32)

    def pivot_long_value(column: str) -> np.ndarray:
        return long_frame[column].to_numpy(np.float32).reshape(n_candidates, n_rows).T

    context = {
        "base": frame[["id", "well", "fold", "md_since"]].reset_index(drop=True).copy(),
        "candidate_tvt": candidate_values.astype(np.float32),
        "probability": probability,
        "pred_error": pred_error,
        "multiobs_score": pivot_long_value("candidate_multiobs_score"),
        "multiobs_mae": pivot_long_value("candidate_multiobs_mae"),
        "multiobs_ncc": pivot_long_value("candidate_multiobs_ncc"),
    }
    long_out = long_frame[
        ["id", "well", "candidate_name", "candidate_index", "candidate_tvt"]
    ].copy()
    long_out["pred_within10_prob"] = probability.T.reshape(-1).astype(np.float32)
    long_out["pred_abs_error"] = pred_error.T.reshape(-1).astype(np.float32)
    long_out["baseline_multiobs_score"] = context["multiobs_score"].T.reshape(-1).astype(np.float32)
    long_out["baseline_multiobs_mae"] = context["multiobs_mae"].T.reshape(-1).astype(np.float32)
    long_out["baseline_multiobs_ncc"] = context["multiobs_ncc"].T.reshape(-1).astype(np.float32)
    long_out["md_since"] = np.tile(frame["md_since"].to_numpy(np.float32), n_candidates)
    return context, long_out


def build_ml_features(
    context: dict[str, Any], candidates: list[str], config: dict[str, Any]
) -> pd.DataFrame:
    base = context["base"][["id", "well", "fold", "md_since"]].reset_index(drop=True).copy()
    probability = context["probability"]
    pred_error = context["pred_error"]
    candidate_tvt = context["candidate_tvt"]
    multiobs_score = context["multiobs_score"]
    multiobs_mae = context["multiobs_mae"]
    multiobs_ncc = context["multiobs_ncc"]
    likpf_idx = candidates.index("likpf_mean")

    prob_order = np.argsort(-probability, axis=1)
    err_order = np.argsort(pred_error, axis=1)
    prob_sorted = np.take_along_axis(probability, prob_order, axis=1)
    err_sorted = np.take_along_axis(pred_error, err_order, axis=1)
    entropy = -np.sum(
        np.clip(probability, 1e-6, 1.0) * np.log(np.clip(probability, 1e-6, 1.0)), axis=1
    )

    out = base
    out["learned_prob_top1_index"] = prob_order[:, 0].astype(np.int16)
    out["learned_error_top1_index"] = err_order[:, 0].astype(np.int16)
    out["learned_prob_top1_value"] = prob_sorted[:, 0].astype(np.float32)
    out["learned_prob_top2_value"] = prob_sorted[:, 1].astype(np.float32)
    out["learned_prob_margin_top1_top2"] = (prob_sorted[:, 0] - prob_sorted[:, 1]).astype(
        np.float32
    )
    out["learned_prob_entropy"] = entropy.astype(np.float32)
    out["learned_error_top1_value"] = err_sorted[:, 0].astype(np.float32)
    out["learned_error_top2_value"] = err_sorted[:, 1].astype(np.float32)
    out["learned_error_margin_top2_top1"] = (err_sorted[:, 1] - err_sorted[:, 0]).astype(np.float32)
    out["learned_prob_likpf_rank"] = rank_desc(probability)[:, likpf_idx].astype(np.int16)
    out["learned_error_likpf_rank"] = rank_asc(pred_error)[:, likpf_idx].astype(np.int16)
    out["learned_prob_top3_contains_likpf"] = (rank_desc(probability)[:, likpf_idx] < 3).astype(
        np.int8
    )
    out["learned_error_top3_contains_likpf"] = (rank_asc(pred_error)[:, likpf_idx] < 3).astype(
        np.int8
    )
    out["candidate_tvt_std"] = candidate_tvt.std(axis=1).astype(np.float32)
    out["candidate_tvt_range"] = (candidate_tvt.max(axis=1) - candidate_tvt.min(axis=1)).astype(
        np.float32
    )
    prob_sum = probability.sum(axis=1)
    prob_sum = np.where(prob_sum > 1e-6, prob_sum, 1.0)
    out["learned_prob_weighted_tvt"] = (
        np.sum(candidate_tvt * probability, axis=1) / prob_sum
    ).astype(np.float32)
    inv_error_weight = 1.0 / np.maximum(pred_error, 1e-3)
    inv_error_sum = inv_error_weight.sum(axis=1)
    out["learned_error_weighted_tvt"] = (
        np.sum(candidate_tvt * inv_error_weight, axis=1) / inv_error_sum
    ).astype(np.float32)

    include_candidate_tvt = bool(
        get_nested(config, "generator.feature_cache.include_candidate_tvt")
    )
    include_multiobs = bool(get_nested(config, "generator.feature_cache.include_multiobs_scores"))
    for idx, candidate in enumerate(candidates):
        out[f"learned_prob_{candidate}"] = probability[:, idx].astype(np.float32)
        out[f"learned_pred_abs_error_{candidate}"] = pred_error[:, idx].astype(np.float32)
        if include_candidate_tvt:
            out[f"candidate_tvt_{candidate}"] = candidate_tvt[:, idx].astype(np.float32)
        if include_multiobs:
            out[f"multiobs_score_{candidate}"] = multiobs_score[:, idx].astype(np.float32)
            out[f"multiobs_mae_{candidate}"] = multiobs_mae[:, idx].astype(np.float32)
            out[f"multiobs_ncc_{candidate}"] = multiobs_ncc[:, idx].astype(np.float32)
    return out


def generate_ml_features_from_frame(
    frame: pd.DataFrame,
    *,
    candidates: list[CandidateSpec],
    row_feature_columns: list[str],
    model_feature_columns: list[str],
    classifier: Any,
    error_model: Any,
    config: dict[str, Any],
    default_fold: int = -1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame, _engineered, candidate_values = add_target_free_row_features(
        frame,
        candidates,
        include_candidate_values=bool(get_nested(config, "generator.include_candidate_values")),
    )
    if "fold" not in frame.columns:
        frame["fold"] = np.int16(default_fold)
    if "md_since" not in frame.columns:
        extracted = frame["id"].astype(str).str.extract(r"_(\d+)$", expand=False)
        frame["md_since"] = pd.to_numeric(extracted, errors="coerce").fillna(0).astype(np.float32)
    context, long_likelihood = predict_likelihood_matrices(
        frame,
        candidates,
        row_feature_columns=row_feature_columns,
        model_feature_columns=model_feature_columns,
        classifier=classifier,
        error_model=error_model,
        candidate_values=candidate_values,
    )
    ml_features = build_ml_features(
        context,
        [spec.name for spec in candidates],
        config,
    )
    feature_cols = [col for col in ml_features.columns if col not in {"id", "well"}]
    for col in feature_cols:
        ml_features[col] = pd.to_numeric(ml_features[col], errors="coerce").astype(np.float32)
    if not np.isfinite(ml_features[feature_cols].to_numpy(np.float32)).all():
        raise ValueError("generated ML feature frame contains non-finite values")
    return ml_features, long_likelihood


def schema_parity_frame(generated_columns: list[str], reference_columns: list[str]) -> pd.DataFrame:
    rows = []
    max_len = max(len(generated_columns), len(reference_columns))
    for index in range(max_len):
        generated = generated_columns[index] if index < len(generated_columns) else None
        reference = reference_columns[index] if index < len(reference_columns) else None
        rows.append(
            {
                "feature_index": index,
                "generated_feature": generated,
                "reference_feature": reference,
                "matches_position": generated == reference,
                "generated_only": generated is not None and generated not in reference_columns,
                "reference_only": reference is not None and reference not in generated_columns,
            }
        )
    return pd.DataFrame(rows)


def output_schema(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"feature_index": range(len(frame.columns)), "feature": frame.columns})


def write_ml_features(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, compression="gzip")


def generate_from_cache(
    *,
    source: Path,
    output_path: Path,
    candidates: list[CandidateSpec],
    row_feature_columns: list[str],
    model_feature_columns: list[str],
    classifier: Any,
    error_model: Any,
    config: dict[str, Any],
    required_columns: list[str],
    chunksize: int,
    max_rows: int | None,
) -> dict[str, Any]:
    validate_source_header(source, required_columns)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    wells: set[str] = set()
    first = True
    long_sample: pd.DataFrame | None = None
    with gzip.open(output_path, "wt", newline="") as fp:
        reader = pd.read_csv(
            source,
            usecols=required_columns,
            dtype={"id": str, "well": str},
            chunksize=chunksize,
            nrows=max_rows,
            low_memory=False,
        )
        for chunk_index, chunk in enumerate(reader):
            features, long_likelihood = generate_ml_features_from_frame(
                chunk,
                candidates=candidates,
                row_feature_columns=row_feature_columns,
                model_feature_columns=model_feature_columns,
                classifier=classifier,
                error_model=error_model,
                config=config,
            )
            if long_sample is None:
                long_sample = long_likelihood.head(50).copy()
            features.to_csv(fp, index=False, header=first)
            first = False
            rows += int(len(features))
            wells.update(features["well"].astype(str).unique().tolist())
            print(f"[cache chunk {chunk_index}] rows={rows:,} wells={len(wells):,}", flush=True)
    return {
        "path": str(output_path),
        "rows": rows,
        "wells": len(wells),
        "sha256": sha256_path(output_path),
        "decompressed_sha256": sha256_path(output_path, decompressed=True),
        "long_likelihood_sample": long_sample,
    }


def generate_rawtest_frame_from_replay(
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    # public replay helpers are inlined in this notebook.

    output_dir = Path(get_nested(config, "runtime.replay_work_dir") or "/tmp/exp145_replay")
    data_dir = ExperimentPaths().raw_data_dir
    replay = get_nested(config, "generator.rawtest_replay") or {}
    configure_public_runtime(
        data_dir=data_dir,
        output_dir=output_dir,
        n_jobs=int(replay.get("n_jobs") or 8),
        pf_seeds=int(replay.get("pf_seeds") or 128),
        pf_particles=int(replay.get("pf_particles") or 500),
        fast=bool(replay.get("fast", False)),
        use_gpu="cpu",
        n_train_wells=None,
    )
    test_frame, meta = build_replay_test_frame()
    variant = str(
        get_nested(config, "generator.rawtest_replay.variant") or "pixiux_likpf_public_replay"
    )
    feature_columns = feature_columns_for_variant(test_frame, variant)
    meta["feature_columns_for_variant"] = int(len(feature_columns))
    return test_frame, meta


def ensure_candidate_value_columns(
    frame: pd.DataFrame, candidates: list[CandidateSpec]
) -> pd.DataFrame:
    out = frame.copy()
    last_known = pd.to_numeric(out["last_known_tvt"], errors="coerce").to_numpy(np.float32)
    for spec in candidates:
        if spec.column in out.columns:
            continue
        delta_col = f"{spec.name}_d"
        if delta_col in out.columns:
            out[spec.column] = last_known + pd.to_numeric(out[delta_col], errors="coerce").to_numpy(
                np.float32
            )
    return out


def ensure_multiobs_columns(
    frame: pd.DataFrame,
    candidates: list[CandidateSpec],
    *,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    required_multiobs = [
        str(value) for value in get_nested(config, "generator.multiobs_global_columns") or []
    ]
    for spec in candidates:
        for suffix in ["score", "mae", "ncc"]:
            required_multiobs.append(f"multiobs_{suffix}_{spec.name}")
    missing = [column for column in required_multiobs if column not in frame.columns]
    if not missing:
        return frame, None

    # multi-observation helpers are inlined in this notebook.

    out = ensure_candidate_value_columns(frame, candidates)
    candidate_names = [spec.name for spec in candidates]
    missing_candidates = [column for column in candidate_names if column not in out.columns]
    if missing_candidates:
        raise ValueError(f"raw-test frame missing candidate value columns: {missing_candidates}")

    existing_candidates = out[["id", "well", *candidate_names]].copy()
    multiobs_config = get_nested(config, "generator.multi_observation_likelihood") or {}
    multiobs, well_summary = build_multi_observation_candidate_frame(
        out,
        existing_candidates,
        train_dir=ExperimentPaths().test_data_dir,
        candidate_names=candidate_names,
        config=multiobs_config,
    )
    out = out.merge(multiobs, on=["id", "well"], how="left", validate="one_to_one")
    remaining = [column for column in required_multiobs if column not in out.columns]
    if remaining:
        raise ValueError(f"failed to generate raw-test multiobs columns: {remaining}")
    return out, {
        "mode": "generated_exp099_multiobs_from_rawtest_prefix_gr",
        "initial_missing_columns": missing,
        "generated_columns": sorted(set(required_multiobs)),
        "well_summary_rows": int(len(well_summary)),
        "well_summary": to_jsonable(well_summary.to_dict("records")),
    }


def run_generator(
    *,
    output_dir: str | Path,
    mode: str,
    train_cache_path: str | Path | None,
    rawtest_cache_path: str | Path | None,
    exp111_schema_path: str | Path | None,
    exp111_manifest_path: str | Path | None,
    exp112_schema_path: str | Path | None,
    max_rows: int | None,
) -> dict[str, Any]:
    t0 = time.time()
    config = load_config()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = candidate_specs_from_config(config)
    exp111_artifacts = Path(str(get_nested(config, "data.exp111_artifact_dir_local") or ""))
    exp112_artifacts = Path(str(get_nested(config, "data.exp112_artifact_dir_local") or ""))
    exp099_artifacts = Path(str(get_nested(config, "data.exp099_artifact_dir_local") or ""))
    schema_path = find_artifact(
        DEFAULT_EXP111_SCHEMA,
        exp111_schema_path or get_nested(config, "data.exp111_feature_schema"),
        local_dirs=[exp111_artifacts],
    )
    manifest_path = find_artifact(
        DEFAULT_EXP111_MANIFEST,
        exp111_manifest_path or get_nested(config, "data.exp111_model_manifest"),
        local_dirs=[exp111_artifacts],
    )
    reference_schema_path = find_artifact(
        DEFAULT_EXP112_SCHEMA,
        exp112_schema_path or get_nested(config, "data.exp112_feature_schema"),
        local_dirs=[exp112_artifacts],
    )
    row_feature_columns = load_feature_schema(schema_path)
    model_feature_columns = exp111_model_feature_columns(row_feature_columns)
    reference_columns = load_feature_schema(reference_schema_path)
    classifier, error_model, model_meta = load_exp111_models(manifest_path=manifest_path)

    required_columns = source_required_columns(config, candidates)
    chunksize = int(get_nested(config, "generator.chunksize") or 200_000)
    outputs: dict[str, Any] = {}
    generated_columns: list[str] | None = None

    if mode in {"train", "both"}:
        source = find_artifact(
            DEFAULT_EXP099_TRAIN_FEATURES,
            train_cache_path or get_nested(config, "data.exp099_train_feature_cache_local"),
            local_dirs=[exp099_artifacts],
        )
        meta = generate_from_cache(
            source=source,
            output_path=output_dir / f"{OUTPUT_PREFIX}_full_train_ml_features.csv.gz",
            candidates=candidates,
            row_feature_columns=row_feature_columns,
            model_feature_columns=model_feature_columns,
            classifier=classifier,
            error_model=error_model,
            config=config,
            required_columns=required_columns,
            chunksize=chunksize,
            max_rows=max_rows,
        )
        outputs["full_train_ml_features"] = {
            k: v for k, v in meta.items() if k != "long_likelihood_sample"
        }
        if meta["long_likelihood_sample"] is not None:
            sample_path = output_dir / f"{OUTPUT_PREFIX}_full_train_likelihood_long_sample.csv"
            meta["long_likelihood_sample"].to_csv(sample_path, index=False)
            outputs["full_train_likelihood_long_sample"] = str(sample_path)
        generated_columns = pd.read_csv(meta["path"], nrows=0).columns.tolist()

    if mode in {"rawtest", "both"}:
        if rawtest_cache_path is not None:
            test_frame = pd.read_csv(rawtest_cache_path, dtype={"id": str, "well": str})
            rawtest_meta: dict[str, Any] = {
                "source": str(rawtest_cache_path),
                "source_sha256": sha256_path(Path(rawtest_cache_path)),
                "mode": "provided_rawtest_feature_cache",
            }
        else:
            test_frame, rawtest_meta = generate_rawtest_frame_from_replay(config)
        multiobs_meta: dict[str, Any] | None = None
        test_frame, multiobs_meta = ensure_multiobs_columns(
            test_frame,
            candidates,
            config=config,
        )
        test_features, long_likelihood = generate_ml_features_from_frame(
            test_frame[required_columns],
            candidates=candidates,
            row_feature_columns=row_feature_columns,
            model_feature_columns=model_feature_columns,
            classifier=classifier,
            error_model=error_model,
            config=config,
        )
        test_path = output_dir / f"{OUTPUT_PREFIX}_rawtest_ml_features.csv.gz"
        long_path = output_dir / f"{OUTPUT_PREFIX}_rawtest_likelihood_long.csv.gz"
        write_ml_features(test_path, test_features)
        long_likelihood.to_csv(long_path, index=False, compression="gzip")
        outputs["rawtest_ml_features"] = {
            "path": str(test_path),
            "rows": int(len(test_features)),
            "wells": int(test_features["well"].nunique()),
            "sha256": sha256_path(test_path),
            "decompressed_sha256": sha256_path(test_path, decompressed=True),
        }
        outputs["rawtest_likelihood_long"] = {
            "path": str(long_path),
            "rows": int(len(long_likelihood)),
            "sha256": sha256_path(long_path),
            "decompressed_sha256": sha256_path(long_path, decompressed=True),
        }
        outputs["rawtest_source"] = rawtest_meta
        if multiobs_meta is not None:
            outputs["rawtest_multiobs_generation"] = multiobs_meta
        generated_columns = test_features.columns.tolist()

    if generated_columns is None:
        raise ValueError("mode must generate at least one feature file")

    schema_path_out = output_dir / f"{OUTPUT_PREFIX}_feature_schema.csv"
    parity_path = output_dir / f"{OUTPUT_PREFIX}_schema_parity.csv"
    output_schema(pd.DataFrame(columns=generated_columns)).to_csv(schema_path_out, index=False)
    parity = schema_parity_frame(generated_columns, reference_columns)
    parity.to_csv(parity_path, index=False)
    parity_pass = bool(
        len(generated_columns) == len(reference_columns)
        and all(
            left == right for left, right in zip(generated_columns, reference_columns, strict=True)
        )
    )

    summary = {
        "experiment": OUTPUT_PREFIX,
        "status": "implemented_smoke_completed"
        if max_rows is not None
        else "implemented_not_run_full",
        "created_at": datetime.now(UTC).isoformat(),
        "mode": mode,
        "runtime_seconds": float(time.time() - t0),
        "candidates": [spec.name for spec in candidates],
        "row_feature_schema": {
            "path": str(schema_path),
            "sha256": sha256_path(schema_path),
            "feature_count": int(len(row_feature_columns)),
        },
        "exp111_model_feature_order": {
            "feature_count": int(len(model_feature_columns)),
            "features": model_feature_columns,
            "source": "reconstructed_from_exp111_long_feature_columns_order",
        },
        "reference_exp112_schema": {
            "path": str(reference_schema_path),
            "sha256": sha256_path(reference_schema_path),
            "columns": int(len(reference_columns)),
        },
        "generated_schema": {
            "path": str(schema_path_out),
            "sha256": sha256_path(schema_path_out),
            "columns": int(len(generated_columns)),
            "parity_path": str(parity_path),
            "parity_sha256": sha256_path(parity_path),
            "schema_parity_pass": parity_pass,
            "mismatch_rows": int((~parity["matches_position"]).sum()),
        },
        "model": model_meta,
        "outputs": outputs,
        "known_limitations": [
            (
                "exp111 saved fold0 models did not persist the training imputation medians; "
                "this generator imputes per generated batch before LightGBM prediction."
            ),
            "The generator is target-free and does not produce a submission.csv.",
        ],
    }
    summary_path = output_dir / f"{OUTPUT_PREFIX}_summary.json"
    with summary_path.open("w") as fp:
        json.dump(to_jsonable(summary), fp, indent=2, sort_keys=True)
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True), flush=True)
    return summary


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--mode", choices=["train", "rawtest", "both"], default="train")
    parser.add_argument("--train-cache-path", type=Path, default=None)
    parser.add_argument("--rawtest-cache-path", type=Path, default=None)
    parser.add_argument("--exp111-schema-path", type=Path, default=None)
    parser.add_argument("--exp111-manifest-path", type=Path, default=None)
    parser.add_argument("--exp112-schema-path", type=Path, default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args(argv)

    paths = ExperimentPaths()
    config = load_config()
    output_dir = args.output_dir or (
        paths.artifacts_dir
        if not Path("/kaggle/working").exists()
        else Path("/kaggle/working") / "artifacts"
    )
    max_rows = args.max_rows
    configured_max = get_nested(config, "generator.max_rows")
    if max_rows is None and configured_max is not None:
        max_rows = int(configured_max)
    return run_generator(
        output_dir=output_dir,
        mode=args.mode,
        train_cache_path=args.train_cache_path,
        rawtest_cache_path=args.rawtest_cache_path,
        exp111_schema_path=args.exp111_schema_path,
        exp111_manifest_path=args.exp111_manifest_path,
        exp112_schema_path=args.exp112_schema_path,
        max_rows=max_rows,
    )


# Notebook source: keep CLI entry point disabled when helpers are inlined.
# if __name__ == "__main__":
#     main()


# Preserve raw-test helper aliases before exp148 full-train helpers define same names.
find_generator_artifact = find_artifact

# %% [markdown]
# ## 5. Model and inference utilities
#
# Exp148 feature assembly, LightGBM training, saved-booster inference, metrics,
# and artifact writing helpers.

# %%
# Inlined from learned_likelihood_fulltrain_addonly_on_exp092.py
import gzip
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold

EXP072_ARTIFACTS = Path("experiments") / "exp072_exp063_full_replay_feature_cache" / "artifacts"
EXP145_TRAIN_ARTIFACTS = (
    Path("experiments")
    / "exp145_learned_likelihood_rawtest_feature_generator_parity"
    / "kaggle"
    / "output"
    / "train_v2"
    / "artifacts"
)
EXP145_INFERENCE_ARTIFACTS = (
    Path("experiments")
    / "exp145_learned_likelihood_rawtest_feature_generator_parity"
    / "kaggle"
    / "output"
    / "inference_v3"
    / "artifacts"
)
FULL_REPLAY_TRAIN_FEATURES = (
    "exp063_full_replay_feature_cache_pixiux_likpf_public_replay_train_features.csv.gz"
)
FULL_REPLAY_FEATURE_SCHEMA = "exp063_full_replay_feature_cache_feature_schema.csv"
FULL_REPLAY_CACHE_SUMMARY = "exp063_full_replay_feature_cache_summary.json"
EXP145_TRAIN_ML_FEATURES = (
    "exp145_learned_likelihood_rawtest_feature_generator_parity_full_train_ml_features.csv.gz"
)
EXP145_RAWTEST_ML_FEATURES = (
    "exp145_learned_likelihood_rawtest_feature_generator_parity_rawtest_ml_features.csv.gz"
)
EXP145_FEATURE_SCHEMA = (
    "exp145_learned_likelihood_rawtest_feature_generator_parity_feature_schema.csv"
)
EXP145_SUMMARY = "exp145_learned_likelihood_rawtest_feature_generator_parity_summary.json"
OUTPUT_PREFIX = "exp148_learned_likelihood_fulltrain_addonly_on_exp092"
META_COLUMNS = {"id", "well", "target"}
EXPECTED_FULL_REPLAY_FEATURE_COUNT = 196


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(np.asarray(y_true, float), np.asarray(y_pred, float))))


def sha256_file(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def sha256_gzip_decompressed(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with gzip.open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def prediction_sha256(ids: pd.Series, values: np.ndarray, *, label: str) -> str:
    hasher = hashlib.sha256()
    hasher.update(label.encode("utf-8"))
    for raw_id in ids.astype(str).to_numpy():
        hasher.update(raw_id.encode("utf-8"))
        hasher.update(b"\0")
    hasher.update(np.asarray(values, dtype=np.float32).tobytes())
    return hasher.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if pd.isna(value) and not isinstance(value, str):
        return None
    return value


def find_artifact(
    filename: str,
    explicit_path: str | Path | None = None,
    *,
    local_artifacts: Path = EXP072_ARTIFACTS,
) -> Path:
    candidates: list[Path] = []
    if explicit_path is not None:
        candidates.append(Path(explicit_path))
    candidates.extend(
        [
            local_artifacts / filename,
            Path.cwd() / filename,
            Path.cwd() / "artifacts" / filename,
        ]
    )
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        candidates.extend(kaggle_input.glob(f"**/{filename}"))
    for candidate in candidates:
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    checked = "\n".join(str(path) for path in candidates[:80])
    raise FileNotFoundError(f"artifact not found or empty: {filename}. Checked:\n{checked}")


def exp063_lgb_config_family(*, fast: bool = False) -> list[dict[str, Any]]:
    base: dict[str, Any] = {
        "boosting_type": "gbdt",
        "objective": "regression",
        "verbose": -1,
        "max_bin": 255,
    }
    n_estimators = 600 if fast else 5000
    return [
        {
            **base,
            "num_leaves": 255,
            "min_child_samples": 15,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.8,
            "reg_lambda": 3.0,
            "reg_alpha": 0.05,
            "learning_rate": 0.03,
            "n_estimators": n_estimators,
            "seed": 123,
        },
        {
            **base,
            "num_leaves": 64,
            "min_child_samples": 40,
            "subsample": 0.474,
            "subsample_freq": 1,
            "colsample_bytree": 0.393,
            "reg_lambda": 95.75,
            "reg_alpha": 10.79,
            "min_child_weight": 0.24,
            "learning_rate": 0.0093,
            "n_estimators": min(2 * n_estimators, 10000),
            "random_state": 0,
        },
        {
            **base,
            "num_leaves": 64,
            "min_child_samples": 40,
            "subsample": 0.474,
            "subsample_freq": 1,
            "colsample_bytree": 0.393,
            "reg_lambda": 95.75,
            "reg_alpha": 10.79,
            "min_child_weight": 0.24,
            "learning_rate": 0.0093,
            "n_estimators": min(2 * n_estimators, 10000),
            "random_state": 29,
        },
    ]


def apply_mode_overrides(
    configs: list[dict[str, Any]],
    mode_config: dict[str, Any],
) -> list[dict[str, Any]]:
    use_gpu = bool(mode_config.get("use_gpu", False))
    common = dict(mode_config.get("common_overrides") or {})
    updated: list[dict[str, Any]] = []
    for params in configs:
        merged = dict(params)
        if use_gpu:
            merged["device_type"] = "gpu"
        else:
            merged.pop("device_type", None)
            merged.pop("gpu_use_dp", None)
        merged.update(common)
        if use_gpu and "gpu_use_dp" not in merged:
            merged["gpu_use_dp"] = False
        updated.append(merged)
    return updated


def load_exp072_full_replay_cache_frame(
    cache_path: str | Path | None = None,
    *,
    max_rows: int | None = None,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    source = find_artifact(FULL_REPLAY_TRAIN_FEATURES, cache_path)
    frame = pd.read_csv(source, nrows=max_rows, dtype={"id": str, "well": str})
    required = {"id", "well", "target", "last_known_tvt", "z"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing columns: {missing}")
    frame["id"] = frame["id"].astype(str)
    frame["well"] = frame["well"].astype(str)
    feature_columns = [col for col in frame.columns if col not in META_COLUMNS]
    if len(feature_columns) != EXPECTED_FULL_REPLAY_FEATURE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_FULL_REPLAY_FEATURE_COUNT} full replay features, "
            f"got {len(feature_columns)} from {source}"
        )
    for col in ["target", *feature_columns]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").astype(np.float32)
    if not np.isfinite(frame[["target", *feature_columns]].to_numpy(np.float32)).all():
        raise ValueError("exp072 full replay cache contains non-finite numeric values")

    schema_path: Path | None = None
    summary_path: Path | None = None
    try:
        schema_path = find_artifact(FULL_REPLAY_FEATURE_SCHEMA)
    except FileNotFoundError:
        schema_path = None
    try:
        summary_path = find_artifact(FULL_REPLAY_CACHE_SUMMARY)
    except FileNotFoundError:
        summary_path = None
    metadata = {
        "source": str(source),
        "source_sha256": sha256_file(source),
        "source_experiment": "exp072_exp063_full_replay_feature_cache",
        "source_kind": "exp063_full_public_replay_train_feature_cache",
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "features": int(len(feature_columns)),
        "feature_columns": feature_columns,
        "schema": str(schema_path) if schema_path else None,
        "schema_sha256": sha256_file(schema_path) if schema_path else None,
        "summary": str(summary_path) if summary_path else None,
        "summary_sha256": sha256_file(summary_path) if summary_path else None,
    }
    return frame, feature_columns, metadata


def load_known_prefix_anchors(train_dir: str | Path, wells: list[str] | pd.Series) -> pd.DataFrame:
    train_dir = Path(train_dir)
    rows: list[dict[str, Any]] = []
    for well in sorted(set(map(str, wells))):
        path = train_dir / f"{well}__horizontal_well.csv"
        if not path.exists():
            raise FileNotFoundError(f"raw train well file not found for anchor recovery: {path}")
        frame = pd.read_csv(path, usecols=["MD", "Z", "TVT", "TVT_input"])
        known = frame[pd.to_numeric(frame["TVT_input"], errors="coerce").notna()].copy()
        if known.empty:
            raise ValueError(f"No known TVT_input prefix rows for well {well}")
        anchor = known.iloc[-1]
        rows.append(
            {
                "well": well,
                "anchor_md": float(anchor["MD"]),
                "anchor_z0": float(anchor["Z"]),
                "anchor_t0": float(anchor["TVT_input"]),
                "anchor_tvt_true": float(anchor["TVT"]),
                "known_prefix_rows": int(len(known)),
            }
        )
    return pd.DataFrame(rows)


def add_anchor_columns(
    frame: pd.DataFrame,
    train_dir: str | Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    anchors = load_known_prefix_anchors(train_dir, frame["well"])
    merged = frame.merge(anchors, on="well", how="left", validate="many_to_one")
    if merged[["anchor_t0", "anchor_z0", "anchor_md"]].isna().any().any():
        raise ValueError("Anchor merge produced missing prefix anchor values")
    t0_delta = merged["last_known_tvt"].to_numpy(np.float32) - merged["anchor_t0"].to_numpy(
        np.float32
    )
    meta = {
        "anchor_wells": int(len(anchors)),
        "anchor_t0_vs_last_known_abs_max": float(np.max(np.abs(t0_delta))),
        "anchor_t0_vs_last_known_abs_mean": float(np.mean(np.abs(t0_delta))),
        "known_prefix_rows_min": int(anchors["known_prefix_rows"].min()),
        "known_prefix_rows_max": int(anchors["known_prefix_rows"].max()),
    }
    if meta["anchor_t0_vs_last_known_abs_max"] > 0.05:
        raise ValueError(
            "Recovered raw prefix T0 does not match feature cache last_known_tvt; "
            f"max abs diff={meta['anchor_t0_vs_last_known_abs_max']}"
        )
    return merged, meta


def load_inference_prefix_anchors(
    test_dir: str | Path,
    wells: list[str] | pd.Series,
) -> pd.DataFrame:
    test_dir = Path(test_dir)
    rows: list[dict[str, Any]] = []
    for well in sorted(set(map(str, wells))):
        path = test_dir / f"{well}__horizontal_well.csv"
        if not path.exists():
            raise FileNotFoundError(f"raw test well file not found for anchor recovery: {path}")
        frame = pd.read_csv(path, usecols=["MD", "Z", "TVT_input"])
        known = frame[pd.to_numeric(frame["TVT_input"], errors="coerce").notna()].copy()
        if known.empty:
            raise ValueError(f"No known TVT_input prefix rows for test well {well}")
        anchor = known.iloc[-1]
        rows.append(
            {
                "well": well,
                "anchor_md": float(anchor["MD"]),
                "anchor_z0": float(anchor["Z"]),
                "anchor_t0": float(anchor["TVT_input"]),
                "known_prefix_rows": int(len(known)),
            }
        )
    return pd.DataFrame(rows)


def add_inference_anchor_columns(
    frame: pd.DataFrame,
    test_dir: str | Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    anchors = load_inference_prefix_anchors(test_dir, frame["well"])
    merged = frame.merge(anchors, on="well", how="left", validate="many_to_one")
    if merged[["anchor_t0", "anchor_z0", "anchor_md"]].isna().any().any():
        raise ValueError("Inference anchor merge produced missing prefix anchor values")
    t0_delta = merged["last_known_tvt"].to_numpy(np.float32) - merged["anchor_t0"].to_numpy(
        np.float32
    )
    meta = {
        "anchor_wells": int(len(anchors)),
        "anchor_t0_vs_last_known_abs_max": float(np.max(np.abs(t0_delta))),
        "anchor_t0_vs_last_known_abs_mean": float(np.mean(np.abs(t0_delta))),
        "known_prefix_rows_min": int(anchors["known_prefix_rows"].min()),
        "known_prefix_rows_max": int(anchors["known_prefix_rows"].max()),
    }
    if meta["anchor_t0_vs_last_known_abs_max"] > 0.05:
        raise ValueError(
            "Recovered raw test prefix T0 does not match feature last_known_tvt; "
            f"max abs diff={meta['anchor_t0_vs_last_known_abs_max']}"
        )
    return merged, meta


def find_model_manifest(explicit_path: str | Path | None = None) -> Path:
    candidates: list[Path] = []
    if explicit_path is not None:
        path = Path(explicit_path)
        candidates.append(path if path.name == "manifest.json" else path / "manifest.json")
    candidates.extend(
        [
            Path.cwd() / "artifacts" / f"{OUTPUT_PREFIX}_lgb_models" / "manifest.json",
            Path.cwd() / f"{OUTPUT_PREFIX}_lgb_models" / "manifest.json",
            Path("experiments")
            / "exp092_u_projection_correction_disagreement_fullrun"
            / "kaggle"
            / "output"
            / "train"
            / "artifacts"
            / f"{OUTPUT_PREFIX}_lgb_models"
            / "manifest.json",
        ]
    )
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        candidates.extend(kaggle_input.glob(f"**/{OUTPUT_PREFIX}_lgb_models/manifest.json"))
    for candidate in candidates:
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    checked = "\n".join(str(path) for path in candidates[:120])
    raise FileNotFoundError(f"model manifest not found. Checked:\n{checked}")


def _tail_rank(ids: pd.Series) -> np.ndarray:
    extracted = ids.astype(str).str.extract(r"_(\d+)$", expand=False)
    return pd.to_numeric(extracted, errors="coerce").fillna(-1).to_numpy(np.int32)


def _distance_bucket(values: pd.Series | np.ndarray) -> pd.Categorical:
    numeric = pd.to_numeric(values, errors="coerce")
    return pd.cut(
        numeric,
        bins=[-np.inf, 50.0, 100.0, 250.0, 500.0, 1000.0, np.inf],
        labels=["000_050", "050_100", "100_250", "250_500", "500_1000", "1000_plus"],
        include_lowest=True,
    )


def _tail_rank_bucket(ids: pd.Series) -> pd.Categorical:
    ranks = _tail_rank(ids)
    return pd.cut(
        ranks,
        bins=[-np.inf, 99, 249, 499, 999, np.inf],
        labels=["000_099", "100_249", "250_499", "500_999", "1000_plus"],
        include_lowest=True,
    )


def _source_tvt(frame: pd.DataFrame, name: str, spec: dict[str, Any]) -> np.ndarray:
    if spec.get("enabled", True) is False:
        raise ValueError(f"Projection source is disabled: {name}")
    value_column = spec.get("value_column")
    delta_column = spec.get("delta_column")
    if value_column:
        if value_column not in frame.columns:
            raise ValueError(f"Projection source {name} missing value_column={value_column}")
        return frame[value_column].to_numpy(np.float32)
    if delta_column:
        if delta_column not in frame.columns:
            raise ValueError(f"Projection source {name} missing delta_column={delta_column}")
        return frame["last_known_tvt"].to_numpy(np.float32) + frame[delta_column].to_numpy(
            np.float32
        )
    raise ValueError(f"Projection source {name} needs value_column or delta_column")


def _weighted_polyfit_predict(
    x: np.ndarray,
    y: np.ndarray,
    *,
    degree: int,
    robust_iters: int,
    clip_sigma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 2:
        fill = float(np.nanmedian(y[finite])) if finite.any() else 0.0
        pred = np.full(len(y), fill, dtype=np.float32)
        zeros = np.zeros(len(y), dtype=np.float32)
        return pred, zeros, zeros, 0

    x_fit = x[finite]
    y_fit = y[finite]
    x_center = float(np.median(x_fit))
    x_scale = float(np.nanpercentile(x_fit, 95) - np.nanpercentile(x_fit, 5))
    if not np.isfinite(x_scale) or x_scale <= 1e-6:
        x_scale = max(float(np.max(x_fit) - np.min(x_fit)), 1.0)
    x_norm = (x - x_center) / x_scale
    x_fit_norm = x_norm[finite]
    fit_degree = int(min(max(degree, 0), max(finite.sum() - 1, 0)))
    if np.unique(np.round(x_fit_norm, 8)).size <= fit_degree:
        fit_degree = max(int(np.unique(np.round(x_fit_norm, 8)).size) - 1, 0)

    weights = np.ones(len(y_fit), dtype=np.float64)
    coef = np.array([float(np.mean(y_fit))])
    for _ in range(max(int(robust_iters), 1)):
        coef = np.polyfit(x_fit_norm, y_fit, deg=fit_degree, w=weights)
        residual = y_fit - np.polyval(coef, x_fit_norm)
        mad = float(np.median(np.abs(residual - np.median(residual)))) * 1.4826
        if not np.isfinite(mad) or mad <= 1e-6:
            break
        weights = np.minimum(1.0, (float(clip_sigma) * mad) / (np.abs(residual) + 1e-6))

    poly = np.poly1d(coef)
    pred = poly(x_norm)
    deriv1 = np.polyder(poly, 1)(x_norm) / x_scale
    if fit_degree >= 2:
        deriv2 = np.polyder(poly, 2)(x_norm) / (x_scale * x_scale)
    else:
        deriv2 = np.zeros_like(x_norm)
    return (
        pred.astype(np.float32),
        deriv1.astype(np.float32),
        deriv2.astype(np.float32),
        fit_degree,
    )


def build_u_projection_features(
    frame: pd.DataFrame,
    *,
    source_specs: dict[str, dict[str, Any]],
    degree: int = 3,
    robust_iters: int = 3,
    clip_sigma: float = 4.0,
) -> tuple[pd.DataFrame, dict[str, list[str]], pd.DataFrame]:
    enabled_specs = {
        str(name): dict(spec)
        for name, spec in source_specs.items()
        if dict(spec).get("enabled", True)
    }
    if len(enabled_specs) < 2:
        raise ValueError("At least two enabled projection sources are required")

    result = pd.DataFrame({"id": frame["id"].to_numpy(), "well": frame["well"].to_numpy()})
    group_columns: dict[str, list[str]] = {
        "projection_correction": [],
        "projection_shape": [],
        "u_disagreement": [],
    }
    summary_rows: list[dict[str, Any]] = []
    z = frame["z"].to_numpy(np.float32)
    u0 = frame["anchor_t0"].to_numpy(np.float32) + frame["anchor_z0"].to_numpy(np.float32)
    md_since = frame.get("md_since")
    if md_since is None:
        x_all = np.maximum(
            frame["id"].astype(str).str.extract(r"_(\d+)$", expand=False).astype(float).to_numpy(),
            0.0,
        )
    else:
        x_all = pd.to_numeric(md_since, errors="coerce").to_numpy(np.float32)

    source_u_columns: list[str] = []
    source_corr_columns: list[str] = []
    for source_name, spec in enabled_specs.items():
        prefix = f"uproj_{source_name}"
        tvt_source = _source_tvt(frame, source_name, spec)
        source_u = (tvt_source + z - u0).astype(np.float32)
        result[f"{prefix}_u"] = source_u
        source_u_columns.append(f"{prefix}_u")

        poly = np.zeros(len(frame), dtype=np.float32)
        slope = np.zeros(len(frame), dtype=np.float32)
        curvature = np.zeros(len(frame), dtype=np.float32)
        fit_degree = np.zeros(len(frame), dtype=np.int16)
        for _, idx in frame.groupby("well", sort=False).indices.items():
            idx_array = np.asarray(idx, dtype=np.int64)
            pred, deriv1, deriv2, used_degree = _weighted_polyfit_predict(
                x_all[idx_array],
                source_u[idx_array],
                degree=int(spec.get("degree", degree)),
                robust_iters=int(spec.get("robust_iters", robust_iters)),
                clip_sigma=float(spec.get("clip_sigma", clip_sigma)),
            )
            poly[idx_array] = pred
            slope[idx_array] = deriv1
            curvature[idx_array] = deriv2
            fit_degree[idx_array] = int(used_degree)

        resid = (source_u - poly).astype(np.float32)
        corr = (poly - source_u).astype(np.float32)
        abs_resid = np.abs(resid).astype(np.float32)
        mad_by_well = (
            pd.DataFrame({"well": frame["well"], "abs_resid": abs_resid})
            .groupby("well")["abs_resid"]
            .transform("median")
            .to_numpy(np.float32)
        )
        result[f"{prefix}_poly"] = poly
        result[f"{prefix}_resid"] = resid
        result[f"{prefix}_corr"] = corr
        result[f"{prefix}_abs_resid"] = abs_resid
        result[f"{prefix}_resid_mad"] = mad_by_well
        result[f"{prefix}_slope"] = slope
        result[f"{prefix}_curvature"] = curvature
        result[f"{prefix}_fit_degree"] = fit_degree.astype(np.float32)

        group_columns["projection_correction"].extend(
            [
                f"{prefix}_corr",
                f"{prefix}_resid",
                f"{prefix}_abs_resid",
                f"{prefix}_resid_mad",
            ]
        )
        group_columns["projection_shape"].extend(
            [
                f"{prefix}_poly",
                f"{prefix}_slope",
                f"{prefix}_curvature",
                f"{prefix}_fit_degree",
            ]
        )
        source_corr_columns.append(f"{prefix}_corr")
        summary_rows.append(
            {
                "source": source_name,
                "rows": int(len(source_u)),
                "u_mean": float(np.mean(source_u)),
                "u_std": float(np.std(source_u)),
                "abs_resid_mean": float(np.mean(abs_resid)),
                "abs_resid_p95": float(np.quantile(abs_resid, 0.95)),
                "resid_mad_mean": float(np.mean(mad_by_well)),
            }
        )

    source_names = list(enabled_specs)
    for left_i, left in enumerate(source_names):
        for right in source_names[left_i + 1 :]:
            left_col = f"uproj_{left}_u"
            right_col = f"uproj_{right}_u"
            diff_col = f"uproj_diff_{left}_minus_{right}"
            abs_col = f"uproj_absdiff_{left}_{right}"
            result[diff_col] = result[left_col].to_numpy(np.float32) - result[right_col].to_numpy(
                np.float32
            )
            result[abs_col] = np.abs(result[diff_col].to_numpy(np.float32))
            group_columns["u_disagreement"].extend([diff_col, abs_col])

    source_u_matrix = result[source_u_columns].to_numpy(np.float32)
    result["uproj_source_u_std"] = np.std(source_u_matrix, axis=1).astype(np.float32)
    result["uproj_source_u_range"] = (
        np.max(source_u_matrix, axis=1) - np.min(source_u_matrix, axis=1)
    ).astype(np.float32)
    group_columns["u_disagreement"].extend(["uproj_source_u_std", "uproj_source_u_range"])
    if len(source_corr_columns) >= 2:
        corr_matrix = result[source_corr_columns].to_numpy(np.float32)
        result["uproj_corr_std"] = np.std(corr_matrix, axis=1).astype(np.float32)
        result["uproj_corr_range"] = (
            np.max(corr_matrix, axis=1) - np.min(corr_matrix, axis=1)
        ).astype(np.float32)
        group_columns["u_disagreement"].extend(["uproj_corr_std", "uproj_corr_range"])

    numeric_cols = [col for col in result.columns if col not in {"id", "well"}]
    for col in numeric_cols:
        result[col] = pd.to_numeric(result[col], errors="coerce").astype(np.float32)
    if not np.isfinite(result[numeric_cols].to_numpy(np.float32)).all():
        raise ValueError("U-projection feature frame contains non-finite values")
    return result, group_columns, pd.DataFrame(summary_rows)


def load_learned_likelihood_ml_features(
    feature_path: str | Path | None = None,
    schema_path: str | Path | None = None,
    summary_path: str | Path | None = None,
    *,
    feature_filename: str = EXP145_TRAIN_ML_FEATURES,
    local_artifacts: Path = EXP145_TRAIN_ARTIFACTS,
    source_experiment: str = "exp145_learned_likelihood_rawtest_feature_generator_parity",
    source_kind: str = "target_free_full_train_learned_likelihood_ml_features",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    source = find_artifact(feature_filename, feature_path, local_artifacts=local_artifacts)
    frame = pd.read_csv(source, dtype={"id": str, "well": str})
    required = {"id", "well", "fold", "md_since"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing learned likelihood feature columns: {missing}")
    frame["id"] = frame["id"].astype(str)
    frame["well"] = frame["well"].astype(str)
    if frame.duplicated(["id", "well"]).any():
        duplicated = int(frame.duplicated(["id", "well"]).sum())
        raise ValueError(
            f"learned likelihood ML feature cache has duplicated id/well rows: {duplicated}"
        )
    numeric_cols = [col for col in frame.columns if col not in {"id", "well"}]
    for col in numeric_cols:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").astype(np.float32)
    if not np.isfinite(frame[numeric_cols].to_numpy(np.float32)).all():
        raise ValueError("learned likelihood ML feature cache contains non-finite numeric values")

    resolved_schema: Path | None
    resolved_summary: Path | None
    try:
        resolved_schema = find_artifact(
            EXP145_FEATURE_SCHEMA,
            schema_path,
            local_artifacts=local_artifacts,
        )
    except FileNotFoundError:
        resolved_schema = None
    try:
        resolved_summary = find_artifact(
            EXP145_SUMMARY,
            summary_path,
            local_artifacts=local_artifacts,
        )
    except FileNotFoundError:
        resolved_summary = None
    metadata = {
        "source": str(source),
        "source_sha256": sha256_file(source),
        "source_decompressed_sha256": sha256_gzip_decompressed(source),
        "source_experiment": source_experiment,
        "source_kind": source_kind,
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "columns": int(len(frame.columns)),
        "numeric_columns": numeric_cols,
        "schema": str(resolved_schema) if resolved_schema else None,
        "schema_sha256": sha256_file(resolved_schema) if resolved_schema else None,
        "summary": str(resolved_summary) if resolved_summary else None,
        "summary_sha256": sha256_file(resolved_summary) if resolved_summary else None,
    }
    return frame, metadata


def generate_current_test_learned_likelihood_ml_features(
    *,
    test_frame: pd.DataFrame,
    output_dir: str | Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    # raw-test learned likelihood helpers and settings are inlined in this notebook.

    config = load_config()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = candidate_specs_from_config(config)
    exp111_artifacts = Path(str(get_nested(config, "data.exp111_artifact_dir_local") or ""))
    schema_path = find_generator_artifact(
        DEFAULT_EXP111_SCHEMA,
        get_nested(config, "data.exp111_feature_schema"),
        local_dirs=[exp111_artifacts],
    )
    manifest_path = find_generator_artifact(
        DEFAULT_EXP111_MANIFEST,
        get_nested(config, "data.exp111_model_manifest"),
        local_dirs=[exp111_artifacts],
    )
    row_feature_columns = load_feature_schema(schema_path)
    model_feature_columns = exp111_model_feature_columns(row_feature_columns)
    classifier, error_model, model_meta = load_exp111_models(manifest_path=manifest_path)

    source_frame, multiobs_meta = ensure_multiobs_columns(test_frame, candidates, config=config)
    required_columns = source_required_columns(config, candidates)
    missing = [column for column in required_columns if column not in source_frame.columns]
    if missing:
        raise ValueError(f"current test frame missing learned likelihood source columns: {missing}")
    features, long_likelihood = generate_ml_features_from_frame(
        source_frame[required_columns],
        candidates=candidates,
        row_feature_columns=row_feature_columns,
        model_feature_columns=model_feature_columns,
        classifier=classifier,
        error_model=error_model,
        config=config,
    )
    feature_path = output_dir / f"{OUTPUT_PREFIX}_current_test_learned_likelihood_ml_features.csv.gz"
    long_path = output_dir / f"{OUTPUT_PREFIX}_current_test_learned_likelihood_long.csv.gz"
    write_ml_features(feature_path, features)
    long_likelihood.to_csv(long_path, index=False, compression="gzip")
    return features, {
        "source": str(feature_path),
        "source_sha256": sha256_path(feature_path),
        "source_decompressed_sha256": sha256_path(feature_path, decompressed=True),
        "source_experiment": OUTPUT_PREFIX,
        "source_kind": "target_free_current_test_generated_learned_likelihood_ml_features",
        "rows": int(len(features)),
        "wells": int(features["well"].nunique()),
        "columns": int(len(features.columns)),
        "exp111_schema": str(schema_path),
        "exp111_manifest": str(manifest_path),
        "exp111_model_meta": _jsonable(model_meta),
        "multiobs_generation": _jsonable(multiobs_meta),
        "long_likelihood": {
            "path": str(long_path),
            "rows": int(len(long_likelihood)),
            "sha256": sha256_path(long_path),
            "decompressed_sha256": sha256_path(long_path, decompressed=True),
        },
    }


def learned_feature_keys_match(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    left_keys = left[["id", "well"]].astype(str).sort_values(["id", "well"]).reset_index(drop=True)
    right_keys = (
        right[["id", "well"]].astype(str).sort_values(["id", "well"]).reset_index(drop=True)
    )
    return left_keys.equals(right_keys)


def build_learned_likelihood_features(
    learned_source: pd.DataFrame,
    base_frame: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, list[str]], pd.DataFrame]:
    config = config or {}
    prefix = str(config.get("prefix") or "ll_")
    key_cols = ["id", "well"]
    group_columns: dict[str, list[str]] = {"learned_likelihood_confidence": []}

    direct_columns = [str(col) for col in config.get("direct_columns") or []]
    weighted_tvt_columns = [str(col) for col in config.get("weighted_tvt_columns") or []]
    candidate_tvt_columns = [str(col) for col in config.get("candidate_tvt_columns") or []]
    requested = direct_columns + weighted_tvt_columns + candidate_tvt_columns
    missing = [col for col in requested if col not in learned_source.columns]
    if missing:
        raise ValueError(
            f"learned likelihood ML feature cache missing configured columns: {missing}"
        )

    base_lookup = base_frame[key_cols + ["last_known_tvt", "likpf_mean_d"]].copy()
    base_lookup["likpf_mean_tvt"] = (
        base_lookup["last_known_tvt"].to_numpy(np.float32)
        + base_lookup["likpf_mean_d"].to_numpy(np.float32)
    ).astype(np.float32)
    joined = learned_source[key_cols + requested].merge(
        base_lookup,
        on=key_cols,
        how="inner",
        validate="one_to_one",
    )
    if joined.empty:
        raise ValueError(
            "No shared rows between learned likelihood ML features and base feature frame"
        )
    features = joined[key_cols].copy()

    for col in direct_columns:
        out = f"{prefix}{col}"
        features[out] = joined[col].to_numpy(np.float32)
        group_columns["learned_likelihood_confidence"].append(out)

    for col in weighted_tvt_columns + candidate_tvt_columns:
        raw = joined[col].to_numpy(np.float32)
        minus_last = (raw - joined["last_known_tvt"].to_numpy(np.float32)).astype(np.float32)
        minus_likpf = (raw - joined["likpf_mean_tvt"].to_numpy(np.float32)).astype(np.float32)
        out_last = f"{prefix}{col}_minus_last_known_tvt"
        out_likpf = f"{prefix}{col}_minus_likpf_mean_tvt"
        features[out_last] = minus_last
        features[out_likpf] = minus_likpf
        group_columns["learned_likelihood_confidence"].extend([out_last, out_likpf])

    feature_cols = [col for col in features.columns if col not in key_cols]
    for col in feature_cols:
        features[col] = pd.to_numeric(features[col], errors="coerce").astype(np.float32)
    if not np.isfinite(features[feature_cols].to_numpy(np.float32)).all():
        raise ValueError("learned likelihood feature frame contains non-finite values")

    summary = pd.DataFrame(
        [
            {
                "feature_group": "learned_likelihood_confidence",
                "configured_direct_columns": len(direct_columns),
                "configured_weighted_tvt_columns": len(weighted_tvt_columns),
                "configured_candidate_tvt_columns": len(candidate_tvt_columns),
                "generated_features": len(feature_cols),
                "rows": int(len(features)),
                "wells": int(features["well"].nunique()),
            }
        ]
    )
    return features, group_columns, summary


def feature_columns_for_variant(
    base_feature_columns: list[str],
    feature_group_columns: dict[str, list[str]],
    variant: dict[str, Any],
) -> list[str]:
    columns = list(base_feature_columns)
    groups = list(variant.get("feature_groups") or [])
    extra: list[str] = []
    for group in groups:
        if group not in feature_group_columns:
            raise ValueError(f"Unknown feature group for variant {variant}: {group}")
        extra.extend(feature_group_columns[group])
    for col in variant.get("extra_columns") or []:
        extra.append(str(col))
    seen = set(columns)
    for col in extra:
        if col not in seen:
            columns.append(col)
            seen.add(col)
    return columns


def _by_well_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    frame = predictions.copy()
    frame["error_tvt"] = frame["pred_tvt"] - frame["target_tvt"]
    return (
        frame.groupby(["variant", "mode", "model", "well"], as_index=False)
        .agg(
            rows=("id", "size"),
            rmse_tvt=("error_tvt", lambda value: float(np.sqrt(np.mean(np.square(value))))),
            error_mean=("error_tvt", "mean"),
            error_abs_mean=("error_tvt", lambda value: float(np.mean(np.abs(value)))),
        )
        .sort_values(["variant", "mode", "model", "rmse_tvt"], ascending=[True, True, True, False])
    )


def _bucket_metrics(predictions: pd.DataFrame, source_frame: pd.DataFrame) -> pd.DataFrame:
    frame = predictions[["id", "variant", "mode", "model", "target_tvt", "pred_tvt"]].copy()
    context = source_frame[["id"]].copy()
    distance_source = source_frame.get("md_since", pd.Series(np.nan, index=source_frame.index))
    context["distance_bucket"] = _distance_bucket(distance_source)
    context["tail_rank_bucket"] = _tail_rank_bucket(source_frame["id"])
    frame = frame.merge(context, on="id", how="left", validate="many_to_one")
    frame["error_tvt"] = frame["pred_tvt"] - frame["target_tvt"]
    rows: list[pd.DataFrame] = []
    for bucket_col in ["distance_bucket", "tail_rank_bucket"]:
        grouped = (
            frame.groupby(["variant", "mode", "model", bucket_col], observed=True)
            .agg(
                rows=("id", "size"),
                rmse_tvt=("error_tvt", lambda value: float(np.sqrt(np.mean(np.square(value))))),
                error_abs_mean=("error_tvt", lambda value: float(np.mean(np.abs(value)))),
            )
            .reset_index()
            .rename(columns={bucket_col: "bucket"})
        )
        grouped.insert(3, "bucket_family", bucket_col)
        rows.append(grouped)
    return pd.concat(rows, ignore_index=True)


def _fit_one_variant_mode(
    *,
    variant: dict[str, Any],
    mode_name: str,
    mode_config: dict[str, Any],
    frame: pd.DataFrame,
    feature_columns: list[str],
    output_dir: Path,
    n_splits: int,
    fast: bool,
    early_stopping_rounds: int,
    max_train_rows: int | None,
    save_models: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation

    variant_name = str(variant["name"])
    x_matrix = frame[feature_columns].to_numpy(np.float32)
    y = frame["target"].to_numpy(np.float32)
    base = frame["last_known_tvt"].to_numpy(np.float32)
    target_tvt = base + y
    groups = frame["well"].to_numpy()
    configs = apply_mode_overrides(exp063_lgb_config_family(fast=fast), mode_config)
    cv = GroupKFold(n_splits=int(n_splits))
    rng = np.random.default_rng(42)
    metric_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    importance_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    oof_by_model: list[np.ndarray] = []
    model_dir = output_dir / f"{OUTPUT_PREFIX}_lgb_models" / variant_name / mode_name
    if save_models:
        model_dir.mkdir(parents=True, exist_ok=True)

    print(
        json.dumps(
            {
                "variant": variant_name,
                "mode": mode_name,
                "rows": int(len(frame)),
                "features": int(len(feature_columns)),
                "configs": int(len(configs)),
                "use_gpu": bool(mode_config.get("use_gpu", False)),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    for model_index, params in enumerate(configs):
        oof = np.zeros(len(frame), dtype=np.float32)
        splits = cv.split(x_matrix, y, groups=groups)
        for fold, (train_idx, valid_idx) in enumerate(splits):
            if max_train_rows is not None and len(train_idx) > int(max_train_rows):
                train_idx = np.sort(rng.choice(train_idx, size=int(max_train_rows), replace=False))
            model = LGBMRegressor(**params)
            model.fit(
                x_matrix[train_idx],
                y[train_idx],
                eval_set=[(x_matrix[valid_idx], y[valid_idx])],
                eval_metric="rmse",
                callbacks=[
                    early_stopping(int(early_stopping_rounds), verbose=False),
                    log_evaluation(0),
                ],
            )
            best_iter = int(model.best_iteration_ or params.get("n_estimators", 0))
            pred = model.predict(x_matrix[valid_idx], num_iteration=best_iter).astype(np.float32)
            oof[valid_idx] = pred
            pred_tvt = base[valid_idx] + pred
            model_file = None
            model_sha = None
            if save_models:
                model_file = f"{mode_name}__lgb{model_index}__fold{fold}.txt"
                model_path = model_dir / model_file
                model.booster_.save_model(str(model_path), num_iteration=best_iter)
                model_sha = sha256_file(model_path)
            metric_rows.append(
                {
                    "variant": variant_name,
                    "mode": mode_name,
                    "model": f"lgb{model_index}",
                    "fold": int(fold),
                    "rows": int(len(valid_idx)),
                    "train_rows": int(len(train_idx)),
                    "features": int(len(feature_columns)),
                    "feature_groups": ",".join(variant.get("feature_groups") or []),
                    "best_iteration": best_iter,
                    "rmse_tvt": rmse(target_tvt[valid_idx], pred_tvt),
                    "rmse_target": rmse(y[valid_idx], pred),
                    "prediction_sha256": prediction_sha256(
                        frame.iloc[valid_idx]["id"],
                        pred_tvt,
                        label=f"{variant_name}/{mode_name}/lgb{model_index}/fold{fold}/tvt",
                    ),
                    "model_file": model_file,
                    "model_sha256": model_sha,
                }
            )
            for feature, importance in zip(
                feature_columns,
                model.feature_importances_,
                strict=False,
            ):
                importance_rows.append(
                    {
                        "variant": variant_name,
                        "mode": mode_name,
                        "model": f"lgb{model_index}",
                        "fold": int(fold),
                        "feature": feature,
                        "importance": float(importance),
                    }
                )
            if save_models:
                model_rows.append(
                    {
                        "variant": variant_name,
                        "mode": mode_name,
                        "model": f"lgb{model_index}",
                        "model_index": int(model_index),
                        "fold": int(fold),
                        "best_iteration": best_iter,
                        "file": f"{variant_name}/{mode_name}/{model_file}",
                        "sha256": model_sha,
                    }
                )
            print(
                json.dumps(
                    {
                        "variant": variant_name,
                        "mode": mode_name,
                        "model": f"lgb{model_index}",
                        "fold": int(fold),
                        "rmse_tvt": metric_rows[-1]["rmse_tvt"],
                        "best_iteration": best_iter,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        oof_by_model.append(oof)
        pred_tvt = base + oof
        metric_rows.append(
            {
                "variant": variant_name,
                "mode": mode_name,
                "model": f"lgb{model_index}",
                "fold": "pooled",
                "rows": int(len(frame)),
                "train_rows": None,
                "features": int(len(feature_columns)),
                "feature_groups": ",".join(variant.get("feature_groups") or []),
                "best_iteration": None,
                "rmse_tvt": rmse(target_tvt, pred_tvt),
                "rmse_target": rmse(y, oof),
                "prediction_sha256": prediction_sha256(
                    frame["id"],
                    pred_tvt,
                    label=f"{variant_name}/{mode_name}/lgb{model_index}/pooled/tvt",
                ),
                "model_file": None,
                "model_sha256": None,
            }
        )
        prediction_frames.append(
            pd.DataFrame(
                {
                    "id": frame["id"].to_numpy(),
                    "well": frame["well"].to_numpy(),
                    "variant": variant_name,
                    "mode": mode_name,
                    "model": f"lgb{model_index}",
                    "last_known_tvt": base,
                    "target": y,
                    "target_tvt": target_tvt,
                    "pred_target": oof,
                    "pred_tvt": pred_tvt,
                }
            )
        )

    ensemble = np.mean(np.vstack(oof_by_model), axis=0).astype(np.float32)
    ensemble_tvt = base + ensemble
    ensemble_sha = prediction_sha256(
        frame["id"],
        ensemble_tvt,
        label=f"{variant_name}/{mode_name}/lgb_mean/pooled/tvt",
    )
    metric_rows.append(
        {
            "variant": variant_name,
            "mode": mode_name,
            "model": "lgb_mean",
            "fold": "pooled",
            "rows": int(len(frame)),
            "train_rows": None,
            "features": int(len(feature_columns)),
            "feature_groups": ",".join(variant.get("feature_groups") or []),
            "best_iteration": None,
            "rmse_tvt": rmse(target_tvt, ensemble_tvt),
            "rmse_target": rmse(y, ensemble),
            "prediction_sha256": ensemble_sha,
            "model_file": None,
            "model_sha256": None,
        }
    )
    prediction_frames.append(
        pd.DataFrame(
            {
                "id": frame["id"].to_numpy(),
                "well": frame["well"].to_numpy(),
                "variant": variant_name,
                "mode": mode_name,
                "model": "lgb_mean",
                "last_known_tvt": base,
                "target": y,
                "target_tvt": target_tvt,
                "pred_target": ensemble,
                "pred_tvt": ensemble_tvt,
            }
        )
    )
    mode_summary = {
        "variant": variant_name,
        "mode": mode_name,
        "description": mode_config.get("description"),
        "feature_count": int(len(feature_columns)),
        "feature_groups": list(variant.get("feature_groups") or []),
        "use_gpu": bool(mode_config.get("use_gpu", False)),
        "common_overrides": mode_config.get("common_overrides") or {},
        "lgb_configs": configs,
        "lgb_mean_prediction_sha256": ensemble_sha,
        "model_count": int(len(model_rows)),
    }
    return (
        pd.DataFrame(metric_rows),
        pd.concat(prediction_frames, ignore_index=True),
        pd.DataFrame(importance_rows),
        model_rows,
        mode_summary,
    )


def _plot_mean_importance(mean_importance: pd.DataFrame, output_path: Path, top_n: int) -> None:
    import matplotlib.pyplot as plt

    variants = mean_importance["variant"].drop_duplicates().tolist()
    if not variants:
        return
    fig, axes = plt.subplots(
        len(variants),
        1,
        figsize=(12, max(4, 0.28 * int(top_n) * len(variants))),
        squeeze=False,
    )
    for ax, variant in zip(axes.ravel(), variants, strict=False):
        subset = mean_importance[mean_importance["variant"].eq(variant)].nlargest(
            top_n,
            "mean_importance",
        )
        subset = subset.sort_values("mean_importance", ascending=True)
        ax.barh(subset["feature"], subset["mean_importance"], color="#2f6f8f")
        ax.set_title(str(variant))
        ax.set_xlabel("mean feature_importances_")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def run_learned_likelihood_fulltrain_addonly_on_exp092(
    *,
    output_dir: str | Path,
    train_dir: str | Path,
    cache_path: str | Path | None = None,
    learned_feature_path: str | Path | None = None,
    learned_schema_path: str | Path | None = None,
    learned_summary_path: str | Path | None = None,
    projection_config: dict[str, Any] | None = None,
    learned_feature_config: dict[str, Any] | None = None,
    variants: list[dict[str, Any]] | None = None,
    modes: dict[str, dict[str, Any]] | None = None,
    active_modes: list[str] | tuple[str, ...] | None = None,
    n_splits: int = 5,
    fast: bool = False,
    early_stopping_rounds: int = 250,
    max_rows: int | None = None,
    max_train_rows: int | None = None,
    save_models: bool = True,
    save_predictions: bool = True,
    top_n_importance: int = 40,
) -> dict[str, Any]:
    t0 = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame, base_feature_columns, feature_meta = load_exp072_full_replay_cache_frame(
        cache_path,
        max_rows=max_rows,
    )
    frame, anchor_meta = add_anchor_columns(frame, train_dir)
    learned_features_source, learned_source_meta = load_learned_likelihood_ml_features(
        learned_feature_path,
        schema_path=learned_schema_path,
        summary_path=learned_summary_path,
    )

    projection_config = projection_config or {}
    if projection_config.get("include_lgb_oof_features", False):
        raise NotImplementedError(
            "LGB OOF U-projection features require nested fold generation. "
            "This first ablation keeps them disabled to avoid leakage."
        )
    projection_features, projection_group_columns, projection_summary = build_u_projection_features(
        frame,
        source_specs=dict(projection_config.get("sources") or {}),
        degree=int(projection_config.get("degree", 3)),
        robust_iters=int(projection_config.get("robust_iters", 3)),
        clip_sigma=float(projection_config.get("clip_sigma", 4.0)),
    )
    projection_feature_columns = [
        col for col in projection_features.columns if col not in {"id", "well"}
    ]
    full_frame = pd.concat(
        [
            frame.reset_index(drop=True),
            projection_features[projection_feature_columns].reset_index(drop=True),
        ],
        axis=1,
    )
    learned_features, learned_group_columns, learned_summary = build_learned_likelihood_features(
        learned_features_source,
        full_frame,
        learned_feature_config or {},
    )
    learned_feature_columns = [col for col in learned_features.columns if col not in {"id", "well"}]
    before_rows = len(full_frame)
    before_wells = int(full_frame["well"].nunique())
    full_frame = full_frame.merge(
        learned_features,
        on=["id", "well"],
        how="inner",
        validate="one_to_one",
    )
    if full_frame.empty:
        raise ValueError(
            "No shared rows between exp072/exp092 feature surface and learned likelihood features"
        )
    coverage_meta = {
        "base_rows_before_feature_join": int(before_rows),
        "base_wells_before_feature_join": int(before_wells),
        "learned_feature_rows": int(learned_source_meta["rows"]),
        "learned_feature_wells": int(learned_source_meta["wells"]),
        "joined_rows": int(len(full_frame)),
        "joined_wells": int(full_frame["well"].nunique()),
        "dropped_base_rows": int(before_rows - len(full_frame)),
        "dropped_base_wells": int(before_wells - full_frame["well"].nunique()),
        "full_train_coverage_pass": bool(
            before_rows == len(full_frame) and before_wells == full_frame["well"].nunique()
        ),
    }
    feature_group_columns = {
        **projection_group_columns,
        **learned_group_columns,
    }
    projection_summary.to_csv(
        output_dir / f"{OUTPUT_PREFIX}_projection_feature_summary.csv",
        index=False,
    )
    learned_summary.to_csv(
        output_dir / f"{OUTPUT_PREFIX}_learned_feature_summary.csv",
        index=False,
    )

    selected_variants = list(variants or [])
    if not selected_variants:
        raise ValueError("No feature ablation variants configured")
    variant_names = [str(variant.get("name")) for variant in selected_variants]
    if len(set(variant_names)) != len(variant_names):
        raise ValueError(f"Duplicate variant names: {variant_names}")
    mode_map = modes or {}
    selected_modes = list(active_modes or mode_map)
    if not selected_modes:
        raise ValueError("No active LightGBM modes configured")

    metric_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    importance_frames: list[pd.DataFrame] = []
    model_rows: list[dict[str, Any]] = []
    mode_summaries: list[dict[str, Any]] = []
    feature_schema_rows: list[dict[str, Any]] = []
    for variant in selected_variants:
        if not variant.get("enabled", True):
            continue
        variant_name = str(variant["name"])
        feature_columns = feature_columns_for_variant(
            base_feature_columns,
            feature_group_columns,
            variant,
        )
        for index, feature in enumerate(feature_columns):
            feature_schema_rows.append(
                {
                    "variant": variant_name,
                    "feature_index": int(index),
                    "feature": feature,
                    "is_projection_feature": bool(feature in projection_feature_columns),
                    "is_learned_likelihood_feature": bool(feature in learned_feature_columns),
                }
            )
        for mode_name in selected_modes:
            if mode_name not in mode_map:
                raise ValueError(
                    f"active mode is not defined under model.training.modes: {mode_name}"
                )
            metrics, predictions, importance, models, mode_summary = _fit_one_variant_mode(
                variant=variant,
                mode_name=mode_name,
                mode_config=mode_map[mode_name],
                frame=full_frame,
                feature_columns=feature_columns,
                output_dir=output_dir,
                n_splits=n_splits,
                fast=fast,
                early_stopping_rounds=early_stopping_rounds,
                max_train_rows=max_train_rows,
                save_models=save_models,
            )
            metric_frames.append(metrics)
            prediction_frames.append(predictions)
            importance_frames.append(importance)
            model_rows.extend(models)
            mode_summaries.append(mode_summary)

    metrics = pd.concat(metric_frames, ignore_index=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    importance = pd.concat(importance_frames, ignore_index=True)
    mean_importance = (
        importance.groupby(["variant", "mode", "feature"], as_index=False)
        .agg(
            mean_importance=("importance", "mean"),
            std_importance=("importance", "std"),
            fold_model_records=("importance", "size"),
        )
        .sort_values(["variant", "mode", "mean_importance"], ascending=[True, True, False])
    )
    by_well = _by_well_metrics(predictions)
    bucket_metrics = _bucket_metrics(predictions, full_frame)

    metrics.to_csv(output_dir / f"{OUTPUT_PREFIX}_metrics.csv", index=False)
    by_well.to_csv(output_dir / f"{OUTPUT_PREFIX}_by_well.csv", index=False)
    bucket_metrics.to_csv(output_dir / f"{OUTPUT_PREFIX}_bucket_metrics.csv", index=False)
    importance.to_csv(output_dir / f"{OUTPUT_PREFIX}_feature_importance.csv", index=False)
    mean_importance.to_csv(
        output_dir / f"{OUTPUT_PREFIX}_feature_importance_mean.csv",
        index=False,
    )
    _plot_mean_importance(
        mean_importance,
        output_dir / f"{OUTPUT_PREFIX}_feature_importance_mean_top.png",
        int(top_n_importance),
    )
    if save_predictions:
        predictions.to_csv(
            output_dir / f"{OUTPUT_PREFIX}_predictions.csv.gz",
            index=False,
            compression="gzip",
        )
    pd.DataFrame(feature_schema_rows).to_csv(
        output_dir / f"{OUTPUT_PREFIX}_feature_schema.csv",
        index=False,
    )

    model_root = output_dir / f"{OUTPUT_PREFIX}_lgb_models"
    model_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "experiment": "exp148_learned_likelihood_fulltrain_addonly_on_exp092",
        "parent": "exp092_u_projection_correction_disagreement_fullrun",
        "learned_likelihood_parent": "exp145_learned_likelihood_rawtest_feature_generator_parity",
        "cache_parent": "exp072_exp063_full_replay_feature_cache",
        "mode": "learned_likelihood_fulltrain_addonly_on_exp092_full_train_rows",
        "feature_source": feature_meta,
        "learned_likelihood_feature_source": learned_source_meta,
        "feature_join_coverage": coverage_meta,
        "anchor_source": {
            "train_dir": str(train_dir),
            **anchor_meta,
        },
        "projection_config": projection_config,
        "learned_feature_config": learned_feature_config or {},
        "projection_feature_groups": projection_group_columns,
        "learned_feature_groups": learned_group_columns,
        "n_splits": int(n_splits),
        "variants": selected_variants,
        "models": model_rows,
        "model_count": int(len(model_rows)),
        "modes": mode_summaries,
    }
    (model_root / "manifest.json").write_text(json.dumps(manifest, indent=2))

    pooled = metrics[metrics["fold"].astype(str).eq("pooled")].copy()
    lgb_mean = pooled[pooled["model"].eq("lgb_mean")].sort_values("rmse_tvt")
    best = lgb_mean.iloc[0].to_dict() if not lgb_mean.empty else None
    summary = {
        "experiment": "exp148_learned_likelihood_fulltrain_addonly_on_exp092",
        "status": "train_completed" if not metrics.empty else "implemented_not_run",
        "mode": "learned_likelihood_fulltrain_addonly_on_exp092_full_train_rows",
        "parent": "exp092_u_projection_correction_disagreement_fullrun",
        "learned_likelihood_parent": "exp145_learned_likelihood_rawtest_feature_generator_parity",
        "cache_parent": "exp072_exp063_full_replay_feature_cache",
        "feature_source": feature_meta,
        "learned_likelihood_feature_source": learned_source_meta,
        "feature_join_coverage": coverage_meta,
        "anchor_source": anchor_meta,
        "active_modes": selected_modes,
        "active_variants": variant_names,
        "best_lgb_mean_by_rmse_tvt": _jsonable(best),
        "pooled_metrics": _jsonable(pooled.to_dict("records")),
        "artifacts": {
            "metrics": f"{OUTPUT_PREFIX}_metrics.csv",
            "by_well": f"{OUTPUT_PREFIX}_by_well.csv",
            "bucket_metrics": f"{OUTPUT_PREFIX}_bucket_metrics.csv",
            "projection_feature_summary": f"{OUTPUT_PREFIX}_projection_feature_summary.csv",
            "learned_feature_summary": f"{OUTPUT_PREFIX}_learned_feature_summary.csv",
            "feature_importance": f"{OUTPUT_PREFIX}_feature_importance.csv",
            "feature_importance_mean": f"{OUTPUT_PREFIX}_feature_importance_mean.csv",
            "feature_importance_plot": f"{OUTPUT_PREFIX}_feature_importance_mean_top.png",
            "predictions": f"{OUTPUT_PREFIX}_predictions.csv.gz" if save_predictions else None,
            "feature_schema": f"{OUTPUT_PREFIX}_feature_schema.csv",
            "model_manifest": f"{OUTPUT_PREFIX}_lgb_models/manifest.json",
        },
        "elapsed_seconds": round(time.time() - t0, 3),
    }
    (output_dir / f"{OUTPUT_PREFIX}_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def run_saved_model_inference(
    *,
    output_dir: str | Path,
    submission_path: str | Path,
    sample_submission_path: str | Path,
    data_dir: str | Path,
    test_dir: str | Path,
    model_manifest_path: str | Path | None = None,
    learned_feature_path: str | Path | None = None,
    learned_schema_path: str | Path | None = None,
    learned_summary_path: str | Path | None = None,
    projection_config: dict[str, Any] | None = None,
    learned_feature_config: dict[str, Any] | None = None,
    variant_name: str = "learned_likelihood_confidence_addonly",
    mode_name: str = "gpu_repro_guard_dp_threads8",
    model_name: str = "lgb1",
    submission_target_column: str = "tvt",
    n_jobs: int | None = None,
    pf_seeds: int | None = None,
    pf_particles: int | None = None,
    fast: bool = False,
    use_gpu: str = "auto",
) -> dict[str, Any]:
    import lightgbm as lgb
    # public replay helpers are inlined in this notebook.

    t0 = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    submission_path = Path(submission_path)
    data_dir = Path(data_dir)
    test_dir = Path(test_dir)
    manifest_path = find_model_manifest(model_manifest_path)
    model_root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text())
    projection_config = projection_config or dict(manifest.get("projection_config") or {})
    if projection_config.get("include_lgb_oof_features", False):
        raise NotImplementedError("LGB OOF U-projection features are disabled for exp092 inference")

    print(f"loading saved LightGBM boosters from {model_root}", flush=True)
    configure_public_runtime(
        data_dir=data_dir,
        output_dir=output_dir,
        n_jobs=n_jobs,
        pf_seeds=pf_seeds,
        pf_particles=pf_particles,
        fast=fast,
        use_gpu=use_gpu,
    )
    base_test_frame, test_meta = build_replay_test_frame()
    base_test_frame["id"] = base_test_frame["id"].astype(str)
    base_test_frame["well"] = base_test_frame["well"].astype(str)
    base_feature_columns = [str(col) for col in manifest["feature_source"]["feature_columns"]]
    missing_base = sorted(set(base_feature_columns) - set(base_test_frame.columns))
    if missing_base:
        raise ValueError(f"raw-test replay frame is missing base features: {missing_base[:40]}")
    anchored_frame, anchor_meta = add_inference_anchor_columns(base_test_frame, test_dir)
    projection_features, projection_group_columns, projection_summary = build_u_projection_features(
        anchored_frame,
        source_specs=dict(projection_config.get("sources") or {}),
        degree=int(projection_config.get("degree", 3)),
        robust_iters=int(projection_config.get("robust_iters", 3)),
        clip_sigma=float(projection_config.get("clip_sigma", 4.0)),
    )
    configured_groups = manifest.get("projection_feature_groups") or {}
    if configured_groups and {
        key: list(value) for key, value in projection_group_columns.items()
    } != {key: list(value) for key, value in configured_groups.items()}:
        raise ValueError("Projection feature groups differ from train manifest")

    variant_configs = {
        str(item["name"]): dict(item)
        for item in manifest.get("variants", [])
        if item.get("enabled", True)
    }
    if variant_name not in variant_configs:
        raise ValueError(f"variant={variant_name} not found in train manifest")
    projection_feature_columns = [
        col for col in projection_features.columns if col not in {"id", "well"}
    ]
    test_frame = pd.concat(
        [
            anchored_frame.reset_index(drop=True),
            projection_features[projection_feature_columns].reset_index(drop=True),
        ],
        axis=1,
    )
    try:
        rawtest_learned_features, rawtest_learned_meta = load_learned_likelihood_ml_features(
            learned_feature_path,
            schema_path=learned_schema_path,
            summary_path=learned_summary_path,
            feature_filename=EXP145_RAWTEST_ML_FEATURES,
            local_artifacts=EXP145_INFERENCE_ARTIFACTS,
            source_kind="target_free_rawtest_learned_likelihood_ml_features",
        )
    except FileNotFoundError:
        rawtest_learned_features, rawtest_learned_meta = (
            generate_current_test_learned_likelihood_ml_features(
                test_frame=anchored_frame,
                output_dir=output_dir,
            )
        )
    else:
        if not learned_feature_keys_match(rawtest_learned_features, anchored_frame):
            rawtest_learned_features, rawtest_learned_meta = (
                generate_current_test_learned_likelihood_ml_features(
                    test_frame=anchored_frame,
                    output_dir=output_dir,
                )
            )
    learned_features, learned_group_columns, learned_summary = build_learned_likelihood_features(
        rawtest_learned_features,
        test_frame,
        learned_feature_config or dict(manifest.get("learned_feature_config") or {}),
    )
    learned_feature_columns = [col for col in learned_features.columns if col not in {"id", "well"}]
    before_join_rows = len(test_frame)
    test_frame = test_frame.merge(
        learned_features,
        on=["id", "well"],
        how="inner",
        validate="one_to_one",
    )
    if len(test_frame) != before_join_rows:
        raise ValueError(
            "Raw-test learned likelihood features do not cover every replay test row: "
            f"{len(test_frame)} of {before_join_rows}"
        )
    feature_group_columns = {
        **projection_group_columns,
        **learned_group_columns,
    }
    configured_learned_groups = manifest.get("learned_feature_groups") or {}
    if configured_learned_groups and {
        key: list(value) for key, value in learned_group_columns.items()
    } != {key: list(value) for key, value in configured_learned_groups.items()}:
        raise ValueError("Learned likelihood feature groups differ from train manifest")
    feature_columns = feature_columns_for_variant(
        base_feature_columns,
        feature_group_columns,
        variant_configs[variant_name],
    )

    missing_model = sorted(set(feature_columns) - set(test_frame.columns))
    if missing_model:
        raise ValueError(f"test frame is missing model features: {missing_model[:40]}")
    for col in feature_columns:
        test_frame[col] = pd.to_numeric(test_frame[col], errors="raise").astype(np.float32)
    if not np.isfinite(test_frame[feature_columns].to_numpy(np.float32)).all():
        raise ValueError("test feature matrix contains non-finite values")

    model_rows = [
        item
        for item in manifest.get("models", [])
        if str(item.get("variant")) == variant_name
        and str(item.get("mode")) == mode_name
        and (model_name == "lgb_mean" or str(item.get("model")) == model_name)
    ]
    if not model_rows:
        raise ValueError(
            f"No saved models for variant={variant_name} mode={mode_name} model={model_name}"
        )

    x_matrix = test_frame[feature_columns].to_numpy(np.float32)
    pred_delta = np.zeros(len(test_frame), dtype=np.float32)
    loaded_rows: list[dict[str, Any]] = []
    for item in model_rows:
        model_file = model_root / str(item["file"])
        booster = lgb.Booster(model_file=str(model_file))
        pred = booster.predict(x_matrix).astype(np.float32)
        pred_delta += pred / float(len(model_rows))
        loaded_rows.append(
            {
                "variant": item.get("variant"),
                "mode": item.get("mode"),
                "model": item.get("model"),
                "fold": item.get("fold"),
                "file": str(item.get("file")),
                "sha256": item.get("sha256"),
                "rows": int(len(pred)),
            }
        )

    base = test_frame["last_known_tvt"].to_numpy(np.float32)
    pred_tvt = (base + pred_delta).astype(np.float32)
    predictions = pd.DataFrame(
        {
            "id": test_frame["id"].to_numpy(),
            "well": test_frame["well"].to_numpy(),
            "variant": variant_name,
            "mode": mode_name,
            "model": model_name,
            "last_known_tvt": base,
            "pred_delta": pred_delta,
            "pred_tvt": pred_tvt,
        }
    )

    sample = pd.read_csv(sample_submission_path, dtype={"id": str})
    target_column = (
        submission_target_column
        if submission_target_column in sample.columns
        else str(sample.columns[1])
    )
    pred_map = dict(zip(predictions["id"].astype(str), predictions["pred_tvt"], strict=False))
    mapped = sample["id"].astype(str).map(pred_map)
    fallback = float(predictions["pred_tvt"].mean())
    missing_mask = mapped.isna()

    predictions_path = output_dir / f"{OUTPUT_PREFIX}_inference_test_predictions.csv.gz"
    projection_summary_path = (
        output_dir / f"{OUTPUT_PREFIX}_inference_projection_feature_summary.csv"
    )
    feature_schema_path = output_dir / f"{OUTPUT_PREFIX}_inference_feature_schema.csv"
    predictions.to_csv(predictions_path, index=False, compression="gzip")
    projection_summary.to_csv(projection_summary_path, index=False)
    learned_summary.to_csv(
        output_dir / f"{OUTPUT_PREFIX}_inference_learned_feature_summary.csv",
        index=False,
    )
    pd.DataFrame(
        [
            {
                "feature_index": int(index),
                "feature": feature,
                "is_projection_feature": bool(feature in projection_feature_columns),
                "is_learned_likelihood_feature": bool(feature in learned_feature_columns),
            }
            for index, feature in enumerate(feature_columns)
        ]
    ).to_csv(feature_schema_path, index=False)

    sample[target_column] = mapped.fillna(fallback).astype("float64")
    sample.to_csv(submission_path, index=False)

    submission_sha = sha256_file(submission_path)
    prediction_sha = prediction_sha256(
        predictions["id"],
        pred_delta,
        label=f"{variant_name}/{mode_name}/{model_name}/test",
    )
    metrics = {
        "variant": variant_name,
        "mode": mode_name,
        "model": model_name,
        "model_count": int(len(model_rows)),
        "feature_count": int(len(feature_columns)),
        "test_rows": int(len(test_frame)),
        "submission_rows": int(len(sample)),
        "predicted_rows": int((~missing_mask).sum()),
        "fallback_rows": int(missing_mask.sum()),
        "prediction_min": float(sample[target_column].min()),
        "prediction_max": float(sample[target_column].max()),
        "prediction_mean": float(sample[target_column].mean()),
        "prediction_std": float(sample[target_column].std()),
        "prediction_sha256": prediction_sha,
        "submission_sha256": submission_sha,
    }
    pd.DataFrame([metrics]).to_csv(
        output_dir / f"{OUTPUT_PREFIX}_inference_metrics.csv",
        index=False,
    )
    summary = {
        "experiment": "exp148_learned_likelihood_fulltrain_addonly_on_exp092",
        "status": "inference_completed",
        "mode": "saved_lgb_booster_inference_with_raw_test_feature_replay",
        "train_manifest": str(manifest_path),
        "test_feature_source": test_meta,
        "rawtest_learned_likelihood_feature_source": rawtest_learned_meta,
        "anchor_source": anchor_meta,
        "learned_feature_groups": learned_group_columns,
        "selected": {
            "variant": variant_name,
            "mode": mode_name,
            "model": model_name,
            "model_count": int(len(model_rows)),
        },
        "metrics": metrics,
        "loaded_models": loaded_rows,
        "artifacts": {
            "predictions": predictions_path.name,
            "projection_feature_summary": projection_summary_path.name,
            "learned_feature_summary": f"{OUTPUT_PREFIX}_inference_learned_feature_summary.csv",
            "feature_schema": feature_schema_path.name,
            "metrics": f"{OUTPUT_PREFIX}_inference_metrics.csv",
            "summary": f"{OUTPUT_PREFIX}_inference_summary.json",
            "submission": str(submission_path),
        },
        "known_followup_risk": "OOF worst-well degradation risk remains unresolved.",
        "elapsed_seconds": round(time.time() - t0, 3),
    }
    (output_dir / f"{OUTPUT_PREFIX}_inference_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(json.dumps(summary, indent=2), flush=True)
    return summary

# %% [markdown]
# ## 6. Setup and configuration

# %%
import json
from pathlib import Path

import pandas as pd

def cfg_get(config, dotted_key, default=None):
    value = get_nested(config, dotted_key)
    return default if value is None else value

paths = ExperimentPaths()
paths.require_kaggle_runtime()
paths.ensure_output_dirs()
config = load_config()

active_variants = [v for v in cfg_get(config, "model.feature_ablation.active_variants", []) if v.get("enabled", True)]
active_modes = cfg_get(config, "model.training.active_modes", [])
lgb_config_count = 3
n_folds = int(cfg_get(config, "validation.n_folds", 5))
booster_count = len(active_variants) * len(active_modes) * lgb_config_count * n_folds

print("Experiment:", EXPERIMENT_NAME)
print("Route:", cfg_get(config, "experiment.route"))
print("Mode:", cfg_get(config, "audit.mode"))
print("Parent:", cfg_get(config, "lineage.parent"))
print("Learned likelihood parent:", cfg_get(config, "lineage.learned_likelihood_parent"))
print("Kernel sources:", cfg_get(config, "runtime.kaggle.kernel_sources"))
print("Active modes:", active_modes)
print("Active variants:", [v["name"] for v in active_variants])
print("Planned LightGBM configs:", lgb_config_count, "folds:", n_folds, "boosters:", booster_count)

# %% [markdown]
# ## 7. Input and full-train coverage contract

# %%
cache_path = find_artifact(
    FULL_REPLAY_TRAIN_FEATURES,
    cfg_get(config, "data.exp072_train_feature_cache_local"),
)
learned_path = find_artifact(
    EXP145_TRAIN_ML_FEATURES,
    cfg_get(config, "data.learned_likelihood_train_features_local"),
)
print("exp072 full replay train cache:", cache_path)
print("exp145 full-train learned likelihood feature cache:", learned_path)

base_preview = pd.read_csv(cache_path, nrows=5, dtype={"id": str, "well": str})
learned_preview, learned_meta = load_learned_likelihood_ml_features(
    cfg_get(config, "data.learned_likelihood_train_features_local"),
    schema_path=cfg_get(config, "data.learned_likelihood_train_feature_schema_local"),
    summary_path=cfg_get(config, "data.learned_likelihood_train_summary_local"),
)
print("learned feature rows:", learned_meta["rows"], "wells:", learned_meta["wells"], "columns:", learned_meta["columns"])
preview_cols = [
    c
    for c in [
        "id",
        "well",
        "target",
        "last_known_tvt",
        "z",
        "md_since",
        "pf_ancc",
        "likpf_mean_d",
    ]
    if c in base_preview.columns
]
display(base_preview[preview_cols])
display(learned_preview.head()[[
    "id",
    "well",
    "fold",
    "md_since",
    "learned_prob_top1_value",
    "learned_prob_entropy",
    "learned_pred_abs_error_likpf_mean",
    "candidate_tvt_likpf_mean",
]])

# %% [markdown]
# ## 8. Train full-row control and learned-feature variants

# %%
summary = run_learned_likelihood_fulltrain_addonly_on_exp092(
    output_dir=paths.artifacts_dir,
    train_dir=paths.train_data_dir,
    cache_path=cfg_get(config, "data.exp072_train_feature_cache_local"),
    learned_feature_path=cfg_get(config, "data.learned_likelihood_train_features_local"),
    learned_schema_path=cfg_get(config, "data.learned_likelihood_train_feature_schema_local"),
    learned_summary_path=cfg_get(config, "data.learned_likelihood_train_summary_local"),
    projection_config=cfg_get(config, "model.u_projection", {}),
    learned_feature_config=cfg_get(config, "model.learned_likelihood_features", {}),
    variants=cfg_get(config, "model.feature_ablation.active_variants", []),
    modes=cfg_get(config, "model.training.modes", {}),
    active_modes=cfg_get(config, "model.training.active_modes", []),
    n_splits=int(cfg_get(config, "validation.n_folds", 5)),
    fast=bool(cfg_get(config, "audit.fast", False)),
    early_stopping_rounds=int(cfg_get(config, "model.training.early_stopping_rounds", 250)),
    max_rows=cfg_get(config, "model.training.max_rows"),
    max_train_rows=cfg_get(config, "model.training.max_train_rows"),
    save_models=bool(cfg_get(config, "model.training.save_models", True)),
    save_predictions=bool(cfg_get(config, "model.training.save_predictions", True)),
    top_n_importance=int(cfg_get(config, "model.training.top_n_importance", 60)),
)
print(json.dumps({
    "status": summary["status"],
    "best_lgb_mean_by_rmse_tvt": summary["best_lgb_mean_by_rmse_tvt"],
    "feature_join_coverage": summary["feature_join_coverage"],
}, indent=2))

# %% [markdown]
# ## 9. Metrics and generated artifacts

# %%
metrics = pd.read_csv(paths.artifacts_dir / f"{OUTPUT_PREFIX}_metrics.csv")
by_well = pd.read_csv(paths.artifacts_dir / f"{OUTPUT_PREFIX}_by_well.csv")
bucket_metrics = pd.read_csv(paths.artifacts_dir / f"{OUTPUT_PREFIX}_bucket_metrics.csv")
projection_summary = pd.read_csv(paths.artifacts_dir / f"{OUTPUT_PREFIX}_projection_feature_summary.csv")
learned_summary = pd.read_csv(paths.artifacts_dir / f"{OUTPUT_PREFIX}_learned_feature_summary.csv")
importance_mean = pd.read_csv(paths.artifacts_dir / f"{OUTPUT_PREFIX}_feature_importance_mean.csv")
manifest_path = paths.artifacts_dir / f"{OUTPUT_PREFIX}_lgb_models" / "manifest.json"

pooled = metrics[metrics["fold"].astype(str).eq("pooled")].sort_values("rmse_tvt")
display(pooled)
display(learned_summary)
display(projection_summary)
display(bucket_metrics.head(50))
display(by_well.head(30))
display(importance_mean.head(60))
print("Model manifest:", manifest_path, "exists=", manifest_path.exists())
print("Feature importance plot:", paths.artifacts_dir / f"{OUTPUT_PREFIX}_feature_importance_mean_top.png")
