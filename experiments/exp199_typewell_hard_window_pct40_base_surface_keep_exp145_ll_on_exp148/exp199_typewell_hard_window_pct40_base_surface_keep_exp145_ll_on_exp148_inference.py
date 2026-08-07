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
# # exp199_typewell_hard_window_pct40_base_surface_keep_exp145_ll_on_exp148 inference
#
# Inference is intentionally out of scope for this mixed-provenance diagnostic.
# The clean learned-likelihood regeneration follow-up was removed from the
# backlog after the small exp199 train-side gain.
#

# %% [markdown]
# ## 1. Setup and inference contract

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
print("Inference mode:", cfg_get(config, "inference.mode"))
print("Parent:", cfg_get(config, "lineage.parent"))
print("Base surface parent:", cfg_get(config, "lineage.base_surface_parent"))
print("Learned likelihood parent:", cfg_get(config, "lineage.learned_likelihood_parent"))
print("Direct submit candidate:", False)


# %% [markdown]
# ## 2. Out-of-scope marker

# %%
summary = {
    "experiment": EXPERIMENT_NAME,
    "status": "inference_not_implemented",
    "reason": "mixed_provenance_train_side_diagnostic_only",
    "next_step_if_positive": "none; clean learned-likelihood regeneration follow-up removed from backlog",
}
paths.metrics_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, ensure_ascii=False))
