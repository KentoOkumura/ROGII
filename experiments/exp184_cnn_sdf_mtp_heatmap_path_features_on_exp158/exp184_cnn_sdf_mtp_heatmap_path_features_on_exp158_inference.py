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
# # exp184_cnn_sdf_mtp_heatmap_path_features_on_exp158 inference
#
# This experiment is a train-side selector audit only. It does not create a
# submission candidate, direct heatmap TVT replacement, softmax blend, or
# PF-weight replacement.

# %% [markdown]
# ## Contents
#
# 1. Imports
# 2. Configuration summary
# 3. Train-side audit status

# %% [markdown]
# ## 1. Imports

# %%
from __future__ import annotations

from IPython.display import display

from settings import EXPERIMENT_NAME, ExperimentPaths, get_nested, load_config

# %% [markdown]
# ## 2. Configuration summary

# %%
paths = ExperimentPaths()
config = load_config()
paths.require_kaggle_runtime()
paths.ensure_output_dirs()

print("Experiment:", EXPERIMENT_NAME)
print("Route:", get_nested(config, "experiment.route"))
print("Inference mode:", get_nested(config, "inference.mode"))
print("Selected variant:", get_nested(config, "inference.selected_variant"))
print("Parent:", get_nested(config, "lineage.parent"))
print("Heatmap parent:", get_nested(config, "lineage.heatmap_parent"))
print("Train artifacts dir:", paths.artifacts_dir)

display(
    {
        "status": get_nested(config, "experiment.status"),
        "notes": get_nested(config, "inference.notes"),
        "historical_baselines": get_nested(config, "selector.historical_baselines"),
        "expected_train_artifacts": get_nested(config, "audit.expected_train_artifacts"),
    }
)

# %% [markdown]
# ## 3. Train-side audit status

# %%
print(
    "No inference flow is implemented for exp184. "
    "Run the train notebook first and review global OOF, path switch, worst-well, "
    "near-row, distance bucket, heatmap confidence bucket, and exp115 subgroup metrics "
    "before considering any follow-up."
)
