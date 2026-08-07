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
# # exp161_prefix_crop_window_features_on_exp148 train lgb1
#
# CPU LightGBM split run for config `lgb1` only.

# %% [markdown]
# ## Contents
#
# 1. Setup and configuration
# 2. Input and cache contract
# 3. Train selected LGB config
# 4. Metrics and generated artifacts

# %% [markdown]
# ## 1. Setup and configuration

# %%
from __future__ import annotations

import json

import pandas as pd
from IPython.display import display

from settings import EXPERIMENT_NAME, ExperimentPaths, get_nested, load_config
from prefix_crop_window_features_on_exp148 import (
    EXP145_TRAIN_ML_FEATURES,
    FULL_REPLAY_TRAIN_FEATURES,
    OUTPUT_PREFIX,
    PREFIX_CROP_FEATURE_SUMMARY,
    PREFIX_CROP_TRAIN_FEATURES,
    find_artifact,
    run_prefix_crop_window_features_on_exp148,
)

LGB_CONFIG_INDEX = 1


def cfg_get(config, dotted_key, default=None):
    value = get_nested(config, dotted_key)
    return default if value is None else value


paths = ExperimentPaths()
paths.require_kaggle_runtime()
paths.ensure_output_dirs()
config = load_config()

active_variants = [
    v
    for v in cfg_get(config, "model.feature_ablation.active_variants", [])
    if v.get("enabled", True)
]
active_modes = cfg_get(config, "model.training.active_modes", [])
n_folds = int(cfg_get(config, "validation.n_folds", 5))
booster_count = len(active_variants) * len(active_modes) * n_folds

print("Experiment:", EXPERIMENT_NAME)
print("Split train config:", f"lgb{LGB_CONFIG_INDEX}")
print("Route:", cfg_get(config, "experiment.route"))
print("Parent:", cfg_get(config, "lineage.parent"))
print("Kernel sources:", cfg_get(config, "runtime.kaggle.train_kernel_sources"))
print("Active modes:", active_modes)
print("Active variants:", [v["name"] for v in active_variants])
print("Planned boosters:", booster_count)

# %% [markdown]
# ## 2. Input and cache contract

# %%
cache_path = find_artifact(
    FULL_REPLAY_TRAIN_FEATURES,
    cfg_get(config, "data.exp072_train_feature_cache_local"),
)
learned_path = find_artifact(
    EXP145_TRAIN_ML_FEATURES,
    cfg_get(config, "data.learned_likelihood_train_features_local"),
)
prefix_crop_path = find_artifact(
    PREFIX_CROP_TRAIN_FEATURES,
    cfg_get(config, "data.prefix_crop_train_features_local"),
)
prefix_crop_summary_path = find_artifact(
    PREFIX_CROP_FEATURE_SUMMARY,
    cfg_get(config, "data.prefix_crop_train_summary_local"),
)
prefix_crop_summary = pd.read_csv(prefix_crop_summary_path)
learned_preview = pd.read_csv(learned_path, nrows=5, dtype={"id": str, "well": str})

print("exp072 full replay train cache:", cache_path)
print("exp145 full-train learned likelihood feature cache:", learned_path)
print("prefix crop cache:", prefix_crop_path, "bytes=", prefix_crop_path.stat().st_size)
display(pd.read_csv(cache_path, nrows=5, dtype={"id": str, "well": str}))
display(prefix_crop_summary.head(20))
display(learned_preview.head())

# %% [markdown]
# ## 3. Train selected LGB config

# %%
summary = run_prefix_crop_window_features_on_exp148(
    output_dir=paths.artifacts_dir,
    train_dir=paths.train_data_dir,
    cache_path=cfg_get(config, "data.exp072_train_feature_cache_local"),
    learned_feature_path=cfg_get(config, "data.learned_likelihood_train_features_local"),
    learned_schema_path=cfg_get(config, "data.learned_likelihood_train_feature_schema_local"),
    learned_summary_path=cfg_get(config, "data.learned_likelihood_train_summary_local"),
    projection_config=cfg_get(config, "model.u_projection", {}),
    learned_feature_config=cfg_get(config, "model.learned_likelihood_features", {}),
    prefix_crop_config=cfg_get(config, "model.prefix_crop_window_features", {}),
    prefix_crop_feature_path=cfg_get(config, "data.prefix_crop_train_features_local"),
    prefix_crop_schema_path=cfg_get(config, "data.prefix_crop_train_feature_schema_local"),
    prefix_crop_summary_path=cfg_get(config, "data.prefix_crop_train_summary_local"),
    require_prefix_crop_cache=True,
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
    selected_lgb_config_indices=[LGB_CONFIG_INDEX],
)
print(json.dumps(summary, indent=2))

# %% [markdown]
# ## 4. Metrics and generated artifacts

# %%
metrics = pd.read_csv(paths.artifacts_dir / f"{OUTPUT_PREFIX}_metrics.csv")
importance_mean = pd.read_csv(paths.artifacts_dir / f"{OUTPUT_PREFIX}_feature_importance_mean.csv")
manifest_path = paths.artifacts_dir / f"{OUTPUT_PREFIX}_lgb_models" / "manifest.json"

display(metrics[metrics["fold"].astype(str).eq("pooled")].sort_values("rmse_tvt"))
display(importance_mean.head(60))
print("Model manifest:", manifest_path, "exists=", manifest_path.exists())
