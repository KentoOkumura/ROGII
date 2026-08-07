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
# # exp178 supervised GR window matcher from known TVT prefix inference
#
# This experiment is a train-side diagnostic smoke. It intentionally does not
# produce `submission.csv`.

# %% [markdown]
# ## Contents
# 1. Imports
# 2. Configuration
# 3. Inference decision

# %% [markdown]
# ## 1. Imports

# %%
from __future__ import annotations

from settings import EXPERIMENT_NAME, ExperimentPaths, get_nested, load_config

# %% [markdown]
# ## 2. Configuration

# %%
paths = ExperimentPaths()
config = load_config()

print("Experiment:", EXPERIMENT_NAME)
print("Route:", get_nested(config, "experiment.route"))
print("Status:", get_nested(config, "experiment.status"))
print("Inference mode:", get_nested(config, "inference.mode"))
print("Artifacts:", paths.artifacts_dir)

# %% [markdown]
# ## 3. Inference decision

# %%
print(
    "No inference branch is selected. "
    "exp178 trains a known-prefix GR window match scorer smoke and writes diagnostics only."
)
print("Selected variant:", get_nested(config, "inference.selected_variant"))
print("Submission output: none")
