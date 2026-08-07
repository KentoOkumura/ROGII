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
# # exp207_heatmap_mdn_overlapping_window_path_stitch_probe inference
#
# Diagnostic-only experiment. It intentionally does not create a submission.

# %% [markdown]
# ## Contents
#
# 1. Runtime and configuration
# 2. No-submit guard

# %% [markdown]
# ## 1. Runtime and configuration

# %%
from __future__ import annotations

import json
from datetime import UTC, datetime

from settings import EXPERIMENT_NAME, ExperimentPaths, get_nested, load_config


paths = ExperimentPaths()
paths.require_kaggle_runtime()
paths.ensure_output_dirs()
config = load_config()

print("Experiment:", EXPERIMENT_NAME)
print("Route:", get_nested(config, "experiment.route"))
print("Parent:", get_nested(config, "lineage.parent"))
print("Artifacts:", paths.artifacts_dir)

# %% [markdown]
# ## 2. No-submit guard

# %%
metrics = {
    "experiment": EXPERIMENT_NAME,
    "status": "diagnostic_only_no_inference",
    "created_at": datetime.now(UTC).isoformat(),
    "route": get_nested(config, "experiment.route"),
    "cv": None,
    "public_lb": None,
    "private_lb": None,
    "metric": get_nested(config, "validation.metric"),
    "notes": [
        "exp207 is a train-side cached path stitching diagnostic.",
        "It does not generate test predictions or submission.csv.",
        "Run the train notebook to create stitch readouts.",
    ],
}
paths.metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
print(json.dumps(metrics, indent=2, sort_keys=True))
