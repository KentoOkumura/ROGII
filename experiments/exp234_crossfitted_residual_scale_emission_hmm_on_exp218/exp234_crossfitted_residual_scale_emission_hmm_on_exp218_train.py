# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # exp234_crossfitted_residual_scale_emission_hmm_on_exp218 train
#
# exp218 `lgb_mean` の保存済み OOF を point center に固定し、well 単位で
# cross-fit した residual scale だけを exact HMM の Gaussian emission sigma に使う。
# exp218 の booster / control は再学習しない。scale readout guard が失敗した場合、
# HMM は実行せず、その時点で train-side 不採用として記録する。

# %% [markdown]
# ## Contents
#
# 1. Imports
# 2. Runtime and configuration helpers
# 3. Input contract and planned cost
# 4. Cross-fitted residual-scale audit
# 5. Guarded single-variant HMM audit
# 6. Metrics, diagnostics, and generated artifacts

# %%
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from exact_hmm_smoother import resolve_existing_file
from IPython.display import display
from residual_scale_emission_hmm_audit import run_residual_scale_emission_hmm_audit
from settings import ExperimentPaths, get_nested, load_config

# %% [markdown]
# ## 2. Runtime and configuration helpers
#
# `config.yaml` を唯一の設定入口にする。CPU-only の residual-scale cross-fit と
# HMM はいずれも seed / thread 数を固定し、推論・提出はこの notebook から行わない。

# %%
paths = ExperimentPaths()
paths.ensure_output_dirs()
config = load_config()


def cfg_get(dotted_key: str, default=None):
    value = get_nested(config, dotted_key)
    return default if value is None else value


def locate_candidates(candidates: list[str]) -> Path:
    return resolve_existing_file(paths.root, candidates)


# %% [markdown]
# ## 3. Input contract and planned cost
#
# 先に exp218 OOF point center と exp072 row context の ID coverage を audit helper
# が strict に照合する。ここでは source、route、active variant、CPU cost を notebook
# 上に明示する。

# %%
active_variants = [
    row
    for row in (cfg_get("model.feature_ablation.active_variants", []) or [])
    if bool(row.get("enabled", False))
]
active_modes = list(cfg_get("model.training.active_modes", []) or [])
scale_config = dict(cfg_get("residual_scale", {}) or {})
lgb_emission = dict(cfg_get("lgb_emission", {}) or {})
sources = dict(lgb_emission.get("sources") or {})
center_source_name = str(scale_config["center_source"])
center_source = dict(sources[center_source_name])
context_path = locate_candidates(list(scale_config["context_candidates"]))
center_path = locate_candidates(list(center_source["candidates"]))

planned_cost = {
    "route": cfg_get("experiment.route"),
    "parent": cfg_get("lineage.references", [])[0],
    "active_variants": len(active_variants),
    "residual_scale_groupkfold_fits": int(scale_config["n_splits"]),
    "hmm_variants": (
        len(lgb_emission.get("active_sources") or [])
        * len(lgb_emission.get("lambda_grid") or [])
        * len(lgb_emission.get("sigma_floor_grid") or [])
        * len(lgb_emission.get("sigma_cap_grid") or [])
    ),
    "lightgbm_boosters": 0,
    "parent_or_control_retraining": False,
    "gpu": bool(cfg_get("runtime.kaggle.enable_gpu", False)),
    "numba_num_threads": cfg_get("runtime.numba_num_threads"),
    "inference_enabled": bool(cfg_get("experiment.inference_enabled", False)),
}
print(json.dumps(planned_cost, indent=2, sort_keys=True))
if planned_cost["active_variants"] != 1 or planned_cost["hmm_variants"] != 1:
    raise ValueError("exp234 must keep exactly one active variant and one HMM variant")
if planned_cost["lightgbm_boosters"] != 0 or planned_cost["parent_or_control_retraining"]:
    raise ValueError("exp234 must not retrain exp218 or a control")

print("exp218 OOF source:", center_path)
display(pd.read_csv(center_path, nrows=3))
print("exp072 row-context source:", context_path)
display(pd.read_csv(context_path, nrows=3))


# %% [markdown]
# ## 4. Cross-fitted residual-scale audit
#
# `residual_scale_crossfit.py` は exp218 center の absolute/squared residual を
# target とするが、各 held-out well はその scale model の fit から完全に除外する。
# 出力には fold membership、scale decile の RMSE、sigma floor/cap rate、入力 SHA を残す。

# %%
print("Residual-scale model:")
print(json.dumps(scale_config["estimator"], indent=2, sort_keys=True))
print("Residual-scale guard:")
print(json.dumps(scale_config["guard"], indent=2, sort_keys=True))


# %% [markdown]
# ## 5. Guarded single-variant HMM audit
#
# helper は scale artifact を先に書き、guard が通った場合に限り `lambda=0.50` の
# single HMM variant を実行する。guard failure は正常な終了状態であり、HMM / inference /
# submit を進めない。

# %%
summary = run_residual_scale_emission_hmm_audit()
print("Audit status:", summary["status"])
print("Audit summary:")
print(json.dumps(summary, indent=2, sort_keys=True, default=str))


# %% [markdown]
# ## 6. Metrics, diagnostics, and generated artifacts
#
# 必ず residual-scale calibration と fold separation を確認する。HMM が実行された場合は
# `direct_hmm_comparison.py` が exp218 / exp148 / exp193、distance bucket、hidden-like、
# worst-well、step-delta を同じ artifact directory に書き出す。

# %%
scale_summary = summary["residual_scale_summary"]
scale_guard = scale_summary["guard"]
print("Residual-scale guard passed:", scale_guard["passed"])
print(json.dumps(scale_guard, indent=2, sort_keys=True))

calibration_path = paths.artifacts_dir / scale_summary["outputs"]["calibration"]
folds_path = paths.artifacts_dir / scale_summary["outputs"]["folds"]
display(pd.read_csv(calibration_path))
display(pd.read_csv(folds_path))

if summary["hmm_summary"] is None:
    print("HMM was not run because the pre-HMM residual-scale guard did not pass.")
else:
    comparison = summary["comparison_summary"]
    print("Best HMM candidate:")
    print(json.dumps(comparison["best_hmm_lgb_candidate"], indent=2, sort_keys=True))
    overall_path = Path(comparison["artifacts"]["overall_metrics"])
    display(pd.read_csv(overall_path).head(12))
