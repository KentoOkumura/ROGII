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
# # exp191_typewell_continuity_selector_confidence_replacement_only_on_exp148 train
#
# CPU split training entrypoint. The actual LightGBM runs are split into
# `train_lgb0`, `train_lgb1`, and `train_lgb2` notebooks to reduce timeout risk.

# %% [markdown]
# ## Contents
#
# 1. Setup and configuration
# 2. Input and feature contract
# 3. Split train plan

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

active_variants = [
    variant
    for variant in cfg_get(config, "model.feature_ablation.active_variants", [])
    if variant.get("enabled", True)
]
active_modes = cfg_get(config, "model.training.active_modes", [])
n_folds = int(cfg_get(config, "validation.n_folds", 5))

print("Experiment:", EXPERIMENT_NAME)
print("Route:", cfg_get(config, "experiment.route"))
print("Parent:", cfg_get(config, "lineage.parent"))
print("Continuity selector parent:", cfg_get(config, "lineage.continuity_selector_parent"))
print("Candidate ranker parent:", cfg_get(config, "lineage.candidate_ranker_parent"))
print("Active modes:", active_modes)
print("Active variants:", [variant["name"] for variant in active_variants])
print("Kaggle GPU enabled:", cfg_get(config, "runtime.kaggle.enable_gpu"))

# %% [markdown]
# ## 2. Input and feature contract

# %%
contract = {
    "kept_feature_groups": ["projection_correction", "u_disagreement"],
    "removed_feature_group": "learned_likelihood_confidence",
    "replacement_feature_group": "exp191_continuity_selector_confidence",
    "blocked_features": [
        "raw exp191 selected_tvt",
        "selected_minus_exp148",
        "direct TVT replacement",
        "blend",
        "postprocess",
        "hard gate",
    ],
    "kernel_sources": cfg_get(config, "runtime.kaggle.train_lgb_kernel_sources"),
}
print(json.dumps(contract, indent=2, ensure_ascii=False))

# %% [markdown]
# ## 3. Split train plan

# %%
split_plan = []
for lgb_index in range(3):
    split_plan.append(
        {
            "notebook": f"{EXPERIMENT_NAME}_train_lgb{lgb_index}.ipynb",
            "selected_lgb_config_indices": [lgb_index],
            "boosters": len(active_variants) * len(active_modes) * n_folds,
        }
    )
print(json.dumps(split_plan, indent=2, ensure_ascii=False))
print("Total planned boosters across split notebooks:", sum(item["boosters"] for item in split_plan))
