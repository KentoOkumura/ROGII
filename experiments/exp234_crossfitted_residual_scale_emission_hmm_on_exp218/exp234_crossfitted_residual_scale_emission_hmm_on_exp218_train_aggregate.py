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
# # exp234_crossfitted_residual_scale_emission_hmm_on_exp218 comparison-only readout
#
# Kaggle train v1 が residual scale と exact HMM cache の生成を完了した後、
# hidden-like assignment filename の typo で direct comparison だけが停止した。
# 本 notebook は v1 HMM cache を Kaggle input として再利用し、readout のみを完走する。
# HMM / residual-scale fitting、exp218 control の再学習、inference、submit は実行しない。

# %% [markdown]
# ## Contents
#
# 1. Imports
# 2. Runtime and reuse contract
# 3. Input cache and comparison contract
# 4. Direct comparison readout
# 5. Metrics and generated artifacts

# %%
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from IPython.display import display

from comparison_readout import run_comparison_only_readout
from exact_hmm_smoother import resolve_existing_file
from settings import ExperimentPaths, get_nested, load_config


# %% [markdown]
# ## 2. Runtime and reuse contract

# %%
paths = ExperimentPaths()
paths.ensure_output_dirs()
config = load_config()
comparison_config = dict(get_nested(config, "comparison") or {})
reuse_contract = dict(comparison_config.get("comparison_only") or {})
runtime = dict(get_nested(config, "runtime.kaggle") or {})

print(
    json.dumps(
        {
            "route": get_nested(config, "experiment.route"),
            "source_kernel": reuse_contract.get("source_kernel"),
            "source_kernel_version": reuse_contract.get("source_kernel_version"),
            "hmm_recomputation_allowed": reuse_contract.get("hmm_recomputation_allowed"),
            "lightgbm_boosters": 0,
            "residual_scale_fits": 0,
            "hmm_variants": 0,
            "enable_gpu": runtime.get("enable_gpu"),
            "enable_internet": runtime.get("enable_internet"),
        },
        indent=2,
        sort_keys=True,
    )
)
if bool(reuse_contract.get("hmm_recomputation_allowed", True)):
    raise ValueError("comparison-only package must not allow HMM recomputation")


# %% [markdown]
# ## 3. Input cache and comparison contract
#
# v1 HMM cache は raw/decompressed SHA を config に固定して照合する。exp115 の
# `_from_ppt_` fold assignment と exp218 / exp148 / exp193 OOF は Kaggle kernel sources
# から読む。output cache が不足していれば readout を開始せず停止する。

# %%
cache_path = resolve_existing_file(
    paths.root,
    list(comparison_config.get("hmm_feature_cache") or []),
)
header = pd.read_csv(cache_path, nrows=3)
print("Reused v1 HMM cache:", cache_path)
display(header)
print("Expected source gzip SHA:", reuse_contract.get("source_hmm_cache_gzip_sha256"))
print("Expected content SHA:", reuse_contract.get("source_hmm_cache_decompressed_sha256"))
print("Private cache dataset:", reuse_contract.get("cache_dataset"))
print("Private cache archive candidates:", reuse_contract.get("cache_archive_candidates"))
print("Hidden-like fold assignment candidates:")
print(json.dumps(comparison_config.get("hidden_like", {}), indent=2, sort_keys=True))


# %% [markdown]
# ## 4. Direct comparison readout
#
# `comparison_readout.py` calls only `run_direct_comparison()`. It writes overall,
# distance bucket, hidden-like, by-well, HMM-std calibration, and step-delta artifacts.
# It does not import or invoke residual-scale fitting / HMM generation helpers.

# %%
summary = run_comparison_only_readout()
comparison = summary["comparison_summary"]
print("Readout status:", summary["status"])
print("Best HMM candidate:")
print(json.dumps(comparison["best_hmm_lgb_candidate"], indent=2, sort_keys=True))


# %% [markdown]
# ## 5. Metrics and generated artifacts
#
# %%
overall_path = Path(comparison["artifacts"]["overall_metrics"])
bucket_path = Path(comparison["artifacts"]["distance_bucket_metrics"])
hidden_like_path = Path(comparison["artifacts"]["hidden_like_metrics"])
by_well_path = Path(comparison["artifacts"]["by_well_delta"])

display(pd.read_csv(overall_path).head(12))
display(pd.read_csv(bucket_path).head(24))
display(pd.read_csv(hidden_like_path).head(24))
display(pd.read_csv(by_well_path).sort_values("rmse", ascending=False).head(20))
