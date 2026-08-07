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
# # exp182 cnn sdf mtp heatmap fullfold geometry probe inference
#
# This experiment is a train-side GPU diagnostic only. It intentionally has no
# hidden-test inference, path replacement, softmax blend, or submission path.

# %% [markdown]
# ## Contents
# 1. Setup
# 2. Diagnostic-only guard

# %% [markdown]
# ## 1. Setup

# %%
from __future__ import annotations

from settings import EXPERIMENT_NAME, ExperimentPaths

paths = ExperimentPaths()
paths.require_kaggle_runtime()
paths.ensure_output_dirs()

print("Experiment:", EXPERIMENT_NAME)
print("Artifacts dir:", paths.artifacts_dir)

# %% [markdown]
# ## 2. Diagnostic-only guard

# %%
raise RuntimeError(
    "exp182 is a train-side cnn_sdf_mtp_heatmap_fullfold_geometry_probe diagnostic only. "
    "It does not create submission.csv or run hidden-test inference."
)
