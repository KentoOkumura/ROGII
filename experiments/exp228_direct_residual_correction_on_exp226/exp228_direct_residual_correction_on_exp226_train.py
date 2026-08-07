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
# # exp228_direct_residual_correction_on_exp226 train
#
# This experiment intentionally splits CPU LightGBM training into three notebooks:
# `train_lgb0`, `train_lgb1`, and `train_lgb2`.

# %% [markdown]
# ## Contents
#
# 1. Setup and configuration
# 2. Split training contract
# 3. Expected generated artifacts

# %% [markdown]
# ## 1. Setup and configuration

# %%
from __future__ import annotations

from settings import EXPERIMENT_NAME, ExperimentPaths, get_nested, load_config


def cfg_get(config, dotted_key, default=None):
    value = get_nested(config, dotted_key)
    return default if value is None else value


paths = ExperimentPaths()
config = load_config()

print("Experiment:", EXPERIMENT_NAME)
print("Route:", cfg_get(config, "experiment.route"))
print("Parent:", cfg_get(config, "lineage.parent"))
print("ML feature surface parent:", cfg_get(config, "lineage.ml_feature_surface_parent"))
print("Target:", cfg_get(config, "model.target"))
print("Base prediction:", cfg_get(config, "model.base_prediction"))

# %% [markdown]
# ## 2. Split Training Contract

# %%
active_variants = [
    variant
    for variant in cfg_get(config, "model.feature_ablation.active_variants", [])
    if variant.get("enabled", True)
]
active_modes = cfg_get(config, "model.training.active_modes", [])
n_folds = int(cfg_get(config, "validation.n_folds", 5))
split_lgb_models = cfg_get(config, "model.training.split_lgb_models", {})

print("Active variants:", [variant["name"] for variant in active_variants])
print("Active modes:", active_modes)
print("Split LGB models:", split_lgb_models)
for split_name, selected_models in split_lgb_models.items():
    booster_count = len(active_variants) * len(active_modes) * len(selected_models) * n_folds
    print(split_name, "selected_models=", selected_models, "planned_boosters=", booster_count)

print("Do not push this index notebook for full training.")
print("Prepare and push train_lgb0, train_lgb1, and train_lgb2 instead.")

# %% [markdown]
# ## 3. Expected Generated Artifacts

# %%
print("Output prefix:", cfg_get(config, "audit.output_prefix"))
for artifact in cfg_get(config, "audit.expected_train_artifacts", []):
    print("-", artifact)
print("Artifacts directory:", paths.artifacts_dir)
