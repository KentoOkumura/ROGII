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
# # exp208_heatmap_mdn_dense_stride_window_path_regeneration_probe inference
#
# This experiment is a train-side diagnostic only. It intentionally does not
# create `submission.csv`.

# %% [markdown]
# ## Contents
#
# 1. Imports and runtime setup
# 2. Diagnostic-only no-submit guard

# %% [markdown]
# ## 1. Imports and runtime setup

# %%
from __future__ import annotations

import json

from settings import EXPERIMENT_NAME, ExperimentPaths, get_nested, load_config


paths = ExperimentPaths()
paths.require_kaggle_runtime()
paths.ensure_output_dirs()
config = load_config()

print("Experiment:", EXPERIMENT_NAME)
print("Route:", get_nested(config, "experiment.route"))
print("Status:", get_nested(config, "experiment.status"))
print("Artifacts:", paths.artifacts_dir)

# %% [markdown]
# ## 2. Diagnostic-only no-submit guard

# %%
metrics = {
    "experiment": EXPERIMENT_NAME,
    "status": "inference_not_applicable_diagnostic_only",
    "route": get_nested(config, "experiment.route"),
    "cv": None,
    "public_lb": None,
    "private_lb": None,
    "metric": get_nested(config, "validation.metric"),
    "key_idea": get_nested(config, "lineage.diff_summary"),
    "parent": get_nested(config, "lineage.parent"),
    "notes": [
        "exp208 is a train-side dense path regeneration and stitch diagnostic.",
        "No inference branch, submission.csv, or code submit is in scope.",
    ],
}
paths.metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
print(json.dumps(metrics, indent=2, sort_keys=True))
