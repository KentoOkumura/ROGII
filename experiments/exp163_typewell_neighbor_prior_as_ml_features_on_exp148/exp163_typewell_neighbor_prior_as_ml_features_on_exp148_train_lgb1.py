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
# # exp163_typewell_neighbor_prior_as_ml_features_on_exp148 train_lgb1
#
# CPU split train runner for the `lgb1` LightGBM config. This writes a separate model manifest so inference can load `lgb0`, `lgb1`, and `lgb2` train outputs independently.

# %% [markdown]
# ## 1. Setup and split contract

# %%
from __future__ import annotations

import json

from typewell_neighbor_prior_as_ml_features_on_exp148 import (
    OUTPUT_PREFIX,
    run_typewell_neighbor_prior_as_ml_features_on_exp148,
)
from settings import EXPERIMENT_NAME, ExperimentPaths, get_nested, load_config


SPLIT_LGB_MODEL = "lgb1"
SPLIT_OUTPUT_PREFIX = f"{OUTPUT_PREFIX}_{SPLIT_LGB_MODEL}"


def cfg_get(config, dotted_key, default=None):
    value = get_nested(config, dotted_key)
    return default if value is None else value


paths = ExperimentPaths()
paths.require_kaggle_runtime()
paths.ensure_output_dirs()
config = load_config()

print("Experiment:", EXPERIMENT_NAME)
print("Route:", cfg_get(config, "experiment.route"))
print("Selected split model:", SPLIT_LGB_MODEL)
print("Output prefix:", SPLIT_OUTPUT_PREFIX)
print("Active modes:", cfg_get(config, "model.training.active_modes", []))
print("Kaggle GPU enabled:", cfg_get(config, "runtime.kaggle.enable_gpu"))

# %% [markdown]
# ## 2. CPU LightGBM training

# %%
summary = run_typewell_neighbor_prior_as_ml_features_on_exp148(
    output_dir=paths.artifacts_dir,
    train_dir=paths.train_data_dir,
    cache_path=cfg_get(config, "data.exp072_train_feature_cache_local"),
    learned_feature_path=cfg_get(config, "data.learned_likelihood_train_features_local"),
    learned_schema_path=cfg_get(config, "data.learned_likelihood_train_feature_schema_local"),
    learned_summary_path=cfg_get(config, "data.learned_likelihood_train_summary_local"),
    projection_config=cfg_get(config, "model.u_projection", {}),
    learned_feature_config=cfg_get(config, "model.learned_likelihood_features", {}),
    learned_typewell_prior_config=cfg_get(config, "model.typewell_neighbor_prior", {}),
    variants=cfg_get(config, "model.feature_ablation.active_variants", []),
    modes=cfg_get(config, "model.training.modes", {}),
    active_modes=cfg_get(config, "model.training.active_modes", []),
    n_splits=int(cfg_get(config, "validation.n_folds", 5)),
    fast=bool(cfg_get(config, "audit.fast", False)),
    early_stopping_rounds=int(cfg_get(config, "model.training.early_stopping_rounds", 250)),
    max_rows=cfg_get(config, "model.training.max_rows"),
    max_train_rows=cfg_get(config, "model.training.max_train_rows"),
    save_models=bool(cfg_get(config, "model.training.save_models", True)),
    save_predictions=bool(cfg_get(config, "model.training.save_predictions", True)),
    top_n_importance=int(cfg_get(config, "model.training.top_n_importance", 80)),
    selected_lgb_models=[SPLIT_LGB_MODEL],
    output_prefix=SPLIT_OUTPUT_PREFIX,
)
paths.metrics_path.write_text(
    json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
)
print(json.dumps(summary.get("best_lgb_mean_by_rmse_tvt"), indent=2, ensure_ascii=False))
print("Artifacts:", paths.artifacts_dir)
