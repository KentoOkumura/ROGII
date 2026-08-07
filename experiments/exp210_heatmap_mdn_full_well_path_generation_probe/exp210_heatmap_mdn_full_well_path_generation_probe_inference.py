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
# # exp210_heatmap_mdn_full_well_path_generation_probe inference
#
# This experiment is train-side diagnostic only. It produces selector-facing
# full-well heatmap MDN candidate path artifacts and does not create a
# submission.

# %% [markdown]
# ## Contents
#
# 1. Imports and runtime setup
# 2. Scope guard

# %% [markdown]
# ## 1. Imports and runtime setup

# %%
from __future__ import annotations

import json

from settings import EXPERIMENT_NAME, ExperimentPaths, get_nested, load_config


paths = ExperimentPaths()
paths.require_kaggle_runtime()
config = load_config()

print("Experiment:", EXPERIMENT_NAME)
print("Route:", get_nested(config, "experiment.route"))
print("Status:", get_nested(config, "experiment.status"))
print("Backlog:", get_nested(config, "lineage.backlog"))
print("Dense parent:", get_nested(config, "lineage.dense_parent"))
print("Downstream candidate:", get_nested(config, "lineage.downstream_candidate"))

# %% [markdown]
# ## 2. Scope guard

# %%
scope = {
    "inference_enabled": False,
    "submission_enabled": False,
    "reason": (
        "exp210 is a train-side full-well path artifact contract diagnostic. "
        "It intentionally does not run raw-test heatmap generation, direct TVT "
        "replacement, softmax averaging, PF weight replacement, postprocess "
        "blend, or submission."
    ),
    "train_artifact": (
        "artifacts/exp210_heatmap_mdn_full_well_path_generation_probe_"
        "localtopk10_full_well_candidate_paths.csv.gz"
    ),
}
print(json.dumps(scope, indent=2, sort_keys=True))
