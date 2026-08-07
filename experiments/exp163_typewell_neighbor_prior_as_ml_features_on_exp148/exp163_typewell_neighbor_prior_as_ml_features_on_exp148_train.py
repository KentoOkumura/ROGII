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
# # exp163_typewell_neighbor_prior_as_ml_features_on_exp148 train
#
# This experiment uses split CPU training notebooks to avoid Kaggle timeout:
#
# - `exp163_typewell_neighbor_prior_as_ml_features_on_exp148_train_lgb0.ipynb`
# - `exp163_typewell_neighbor_prior_as_ml_features_on_exp148_train_lgb1.ipynb`
# - `exp163_typewell_neighbor_prior_as_ml_features_on_exp148_train_lgb2.ipynb`
#
# Each notebook trains one LightGBM config over 5 folds and writes an independent manifest.

# %% [markdown]
# ## Split Train Contract

# %%
from __future__ import annotations

from settings import EXPERIMENT_NAME, get_nested, load_config


config = load_config()

print("Experiment:", EXPERIMENT_NAME)
print("Route:", get_nested(config, "experiment.route"))
print("Runtime GPU enabled:", get_nested(config, "runtime.kaggle.enable_gpu"))
print("Active mode:", get_nested(config, "model.training.active_modes"))
print("Active variants:", get_nested(config, "model.feature_ablation.active_variants"))
print("Use the train_lgb0/train_lgb1/train_lgb2 notebooks for execution.")
