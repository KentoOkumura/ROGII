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
# # exp163_typewell_neighbor_prior_as_ml_features_on_exp148 inference
#
# Inference is intentionally not enabled yet. The first stage is a train-side CPU
# add-only feature audit with split `lgb0` / `lgb1` / `lgb2` notebooks. Current-test
# typewell prior feature parity must be designed after OOF results are reviewed.

# %% [markdown]
# ## Status

# %%
from __future__ import annotations

from settings import EXPERIMENT_NAME, get_nested, load_config


config = load_config()

print("Experiment:", EXPERIMENT_NAME)
print("Route:", get_nested(config, "experiment.route"))
print("Inference mode:", get_nested(config, "inference.mode"))
print("Status: inference deferred until split train results and parity review are complete.")
