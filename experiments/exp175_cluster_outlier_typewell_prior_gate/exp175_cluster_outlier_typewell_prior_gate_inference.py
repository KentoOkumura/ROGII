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
# # exp175_cluster_outlier_typewell_prior_gate inference
#
# This experiment is train-side audit only. It intentionally does not create `submission.csv`.

# %% [markdown]
# ## Contents
#
# 1. Imports
# 2. Runtime and configuration check
# 3. No-submission decision

# %% [markdown]
# ## 1. Imports

# %%
from __future__ import annotations

from IPython.display import display

from settings import EXPERIMENT_NAME, ExperimentPaths, get_nested, load_config

# %% [markdown]
# ## 2. Runtime and configuration check

# %%
paths = ExperimentPaths()
config = load_config()
paths.require_kaggle_runtime()
paths.ensure_output_dirs()

print("Experiment:", EXPERIMENT_NAME)
print("Route:", get_nested(config, "experiment.route"))
print("Inference mode:", get_nested(config, "inference.mode"))
print("Selected variant:", get_nested(config, "inference.selected_variant"))
print("Submission path would be:", paths.submission_path)

# %% [markdown]
# ## 3. No-submission decision

# %%
display(
    {
        "status": "no_submission_created",
        "reason": "cluster_outlier_typewell_prior_gate is a train-side posthoc audit only",
        "required_before_inference": [
            "Kaggle train-side OOF result review",
            "max well regression review",
            "exp115 hidden-like stress readout",
            "raw-test/full-train parity design",
        ],
    }
)
