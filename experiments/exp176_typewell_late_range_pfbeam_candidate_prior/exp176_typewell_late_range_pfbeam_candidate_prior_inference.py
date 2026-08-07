# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---
# %% [markdown]
# # exp176_typewell_late_range_pfbeam_candidate_prior inference

# %% [markdown]
# ## Contents
# 1. Imports
# 2. Runtime and configuration
# 3. Inference decision

# %%
from __future__ import annotations

import json

from settings import ExperimentPaths, get_nested, is_kaggle_runtime, load_config

# %% [markdown]
# ## 2. Runtime and configuration

# %%
paths = ExperimentPaths()
config = load_config()
print("experiment:", config["experiment"]["name"])
print("route:", config["experiment"]["route"])
print("kaggle_runtime:", is_kaggle_runtime())
print("experiment_dir:", paths.experiment_dir)

# %% [markdown]
# ## 3. Inference decision

# %%
inference = config.get("inference", {})
print(json.dumps(inference, indent=2, sort_keys=True))
if get_nested(config, "inference.selected_variant") is not None:
    raise RuntimeError("This experiment should not select an inference variant yet.")
print("No inference candidate is selected for exp176. Run train-side audit first.")
