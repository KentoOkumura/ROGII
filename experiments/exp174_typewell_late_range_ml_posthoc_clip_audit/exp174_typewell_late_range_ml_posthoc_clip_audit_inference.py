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
# # exp174_typewell_late_range_ml_posthoc_clip_audit inference
#
# This experiment is a train-side OOF posthoc audit. No inference policy is selected here.

# %% [markdown]
# ## Contents
#
# 1. Setup and configuration
# 2. No-op inference guard

# %% [markdown]
# ## 1. Setup and configuration

# %%
from __future__ import annotations

from settings import EXPERIMENT_NAME, ExperimentPaths, get_nested, load_config

paths = ExperimentPaths()
paths.require_kaggle_runtime()
config = load_config()

print("Experiment:", EXPERIMENT_NAME)
print("Route:", get_nested(config, "experiment.route"))
print("Inference mode:", get_nested(config, "inference.mode"))
print("Selected policy:", get_nested(config, "inference.selected_policy"))

# %% [markdown]
# ## 2. No-op inference guard

# %%
raise RuntimeError(
    "exp174 is a train-side OOF posthoc audit only. "
    "Do not run inference until a policy passes raw-test parity, front-half exception, "
    "and worst-well guard review."
)
