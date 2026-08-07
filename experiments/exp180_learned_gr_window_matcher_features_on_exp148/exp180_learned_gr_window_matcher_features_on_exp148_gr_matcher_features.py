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
# # exp180_learned_gr_window_matcher_features_on_exp148 learned GR matcher features
#
# CPU feature-cache notebook for the train-time learned GR window matcher features.

# %% [markdown]
# ## Contents
#
# 1. Setup and configuration
# 2. Input contract
# 3. Generate learned GR matcher feature cache
# 4. Generated artifacts

# %% [markdown]
# ## 1. Setup and configuration

# %%
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from IPython.display import display

from settings import EXPERIMENT_NAME, ExperimentPaths, get_nested, load_config
from learned_gr_window_matcher_features_on_exp148 import (
    EXP145_TRAIN_ML_FEATURES,
    FULL_REPLAY_TRAIN_FEATURES,
    GR_MATCHER_FEATURE_MANIFEST,
    GR_MATCHER_FEATURE_SCHEMA,
    GR_MATCHER_FEATURE_SUMMARY,
    GR_MATCHER_TRAIN_FEATURES,
    find_artifact,
    generate_gr_matcher_window_feature_cache_on_exp148,
    load_learned_likelihood_ml_features,
)


def cfg_get(config, dotted_key, default=None):
    value = get_nested(config, dotted_key)
    return default if value is None else value


paths = ExperimentPaths()
paths.require_kaggle_runtime()
paths.ensure_output_dirs()
config = load_config()

print("Experiment:", EXPERIMENT_NAME)
print("Route:", cfg_get(config, "experiment.route"))
print("Parent:", cfg_get(config, "lineage.parent"))
print("Learned GR matcher scorer:", cfg_get(config, "model.gr_matcher_window_features.scorer_training"))
print("Output feature cache:", GR_MATCHER_TRAIN_FEATURES)
print("Kernel sources:", cfg_get(config, "runtime.kaggle.kernel_sources"))

# %% [markdown]
# ## 2. Input contract

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
print(
    "learned feature rows:",
    learned_meta["rows"],
    "wells:",
    learned_meta["wells"],
    "columns:",
    learned_meta["columns"],
)
display(base_preview.head())
display(learned_preview.head())

# %% [markdown]
# ## 3. Generate learned GR matcher feature cache

# %%
manifest = generate_gr_matcher_window_feature_cache_on_exp148(
    output_dir=paths.artifacts_dir,
    train_dir=paths.train_data_dir,
    cache_path=cfg_get(config, "data.exp072_train_feature_cache_local"),
    learned_feature_path=cfg_get(config, "data.learned_likelihood_train_features_local"),
    learned_schema_path=cfg_get(config, "data.learned_likelihood_train_feature_schema_local"),
    learned_summary_path=cfg_get(config, "data.learned_likelihood_train_summary_local"),
    projection_config=cfg_get(config, "model.u_projection", {}),
    learned_feature_config=cfg_get(config, "model.learned_likelihood_features", {}),
    gr_matcher_config=cfg_get(config, "model.gr_matcher_window_features", {}),
    max_rows=cfg_get(config, "model.training.max_rows"),
)
print(json.dumps(manifest, indent=2))

# %% [markdown]
# ## 4. Generated artifacts

# %%
feature_path = paths.artifacts_dir / GR_MATCHER_TRAIN_FEATURES
schema_path = paths.artifacts_dir / GR_MATCHER_FEATURE_SCHEMA
summary_path = paths.artifacts_dir / GR_MATCHER_FEATURE_SUMMARY
manifest_path = paths.artifacts_dir / GR_MATCHER_FEATURE_MANIFEST

print("feature cache:", feature_path, "exists=", feature_path.exists())
print("schema:", schema_path, "exists=", schema_path.exists())
print("summary:", summary_path, "exists=", summary_path.exists())
print("manifest:", manifest_path, "exists=", manifest_path.exists())
print("feature cache bytes:", feature_path.stat().st_size if feature_path.exists() else None)

summary = pd.read_csv(summary_path)
schema = pd.read_csv(schema_path)
display(summary)
display(schema.head(80))
