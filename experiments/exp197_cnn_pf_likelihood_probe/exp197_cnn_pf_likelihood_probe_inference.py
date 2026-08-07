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
# # exp197 cnn pf likelihood probe inference
#
# This experiment is train-side diagnostic only. It intentionally does not
# generate raw-test features, `submission.csv`, or a Kaggle submission.

# %% [markdown]
# ## Contents
# 1. Imports
# 2. Configuration guard
# 3. No-inference status

# %% [markdown]
# ## 1. Imports

# %%
from __future__ import annotations

import json

from settings import EXPERIMENT_NAME, get_nested, load_config

# %% [markdown]
# ## 2. Configuration guard

# %%
config = load_config()
print("Experiment:", EXPERIMENT_NAME)
print("Route:", get_nested(config, "experiment.route"))
print("Inference mode:", get_nested(config, "inference.mode"))
print("Selected variant:", get_nested(config, "inference.selected_variant"))

if get_nested(config, "inference.mode") != "not_selected_train_side_diagnostic_only":
    raise RuntimeError("exp197 inference mode must stay disabled until a follow-up is designed.")

# %% [markdown]
# ## 3. No-inference status

# %%
status = {
    "experiment": EXPERIMENT_NAME,
    "status": "no_inference_notebook_by_design",
    "reason": (
        "cnn_pf_likelihood_probe is a train-side frozen candidate scorer audit. "
        "Raw-test parity and worst-well guards are required before any inference branch."
    ),
    "submission_created": False,
}
print(json.dumps(status, indent=2, sort_keys=True))
