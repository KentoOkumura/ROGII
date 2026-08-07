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
# # exp220_row_neighbor_input_context_features_on_exp148 train index
#
# This experiment is CPU-only and splits LightGBM training into three Kaggle
# notebooks: `train_lgb0`, `train_lgb1`, and `train_lgb2`.

# %% [markdown]
# ## Contents
#
# 1. Setup and configuration
# 2. Split training contract
# 3. Feature contract

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
    v
    for v in cfg_get(config, "model.feature_ablation.active_variants", [])
    if v.get("enabled", True)
]
active_modes = cfg_get(config, "model.training.active_modes", [])
n_folds = int(cfg_get(config, "validation.n_folds", 5))
split_indices = cfg_get(config, "model.training.lgb_config_splits", [0, 1, 2])

print("Experiment:", EXPERIMENT_NAME)
print("Route:", cfg_get(config, "experiment.route"))
print("Mode:", cfg_get(config, "audit.mode"))
print("Parent:", cfg_get(config, "lineage.parent"))
print("Comparison parents:", cfg_get(config, "lineage.comparison_parents"))
print("Kernel sources:", cfg_get(config, "runtime.kaggle.train_kernel_sources"))

# %% [markdown]
# ## 2. Split training contract

# %%
planned = {
    "active_variants": [v["name"] for v in active_variants],
    "active_modes": active_modes,
    "lgb_config_splits": split_indices,
    "folds_per_split": n_folds,
    "boosters_per_split": len(active_variants) * len(active_modes) * n_folds,
    "total_boosters": len(active_variants) * len(active_modes) * len(split_indices) * n_folds,
    "control_retraining": False,
    "runtime": "cpu",
    "split_notebooks": [
        f"{EXPERIMENT_NAME}_train_lgb0.ipynb",
        f"{EXPERIMENT_NAME}_train_lgb1.ipynb",
        f"{EXPERIMENT_NAME}_train_lgb2.ipynb",
    ],
}
print(json.dumps(planned, indent=2, sort_keys=True))

# %% [markdown]
# ## 3. Feature contract

# %%
print(json.dumps(cfg_get(config, "model.row_neighbor_input_context_features", {}), indent=2))
print("This index notebook does not fit LightGBM models. Use train_lgb0/1/2.")
