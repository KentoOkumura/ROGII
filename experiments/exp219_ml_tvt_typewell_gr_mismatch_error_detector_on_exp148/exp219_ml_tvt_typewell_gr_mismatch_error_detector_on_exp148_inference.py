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
# # exp219_ml_tvt_typewell_gr_mismatch_error_detector_on_exp148 inference
#
# Initial exp219 implementation is train-side diagnostic only. It does not
# generate a `submission.csv`, saved-booster inference, replacement candidate,
# blend, or postprocess.

# %% [markdown]
# ## Contents
#
# 1. Setup and configuration
# 2. Inference status

# %% [markdown]
# ## 1. Setup and configuration

# %%
from __future__ import annotations

import json

from settings import EXPERIMENT_NAME, ExperimentPaths, get_nested, load_config


def cfg_get(config, dotted_key, default=None):
    value = get_nested(config, dotted_key)
    return default if value is None else value


paths = ExperimentPaths()
paths.require_kaggle_runtime()
paths.ensure_output_dirs()
config = load_config()

print("Experiment:", EXPERIMENT_NAME)
print("Route:", cfg_get(config, "experiment.route"))
print("Status:", cfg_get(config, "experiment.status"))
print("Parent:", cfg_get(config, "lineage.parent"))
print("Inference enabled:", False)
print("Submit enabled:", False)
print("Reason:", cfg_get(config, "audit.inference_status"))

# %% [markdown]
# ## 2. Inference status

# %%
status = {
    "experiment": EXPERIMENT_NAME,
    "inference_enabled": False,
    "submission_generated": False,
    "reason": (
        "exp219 first checks whether ML-TVT GR mismatch features separate "
        "exp148 OOF errors. Current-test feature regeneration and LightGBM "
        "add-only inference are intentionally deferred until the readout gate passes."
    ),
}
print(json.dumps(status, indent=2))
