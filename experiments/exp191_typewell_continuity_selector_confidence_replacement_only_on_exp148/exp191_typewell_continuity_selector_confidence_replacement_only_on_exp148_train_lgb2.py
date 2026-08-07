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
# # exp191_typewell_continuity_selector_confidence_replacement_only_on_exp148 train lgb2
#
# CPU LightGBM split run for config `lgb2` only.

# %% [markdown]
# ## Contents
#
# 1. Setup and configuration
# 2. Input and continuity selector contract
# 3. Train selected LGB config
# 4. Metrics and generated artifacts

# %% [markdown]
# ## 1. Setup and configuration

# %%
from __future__ import annotations

import json

import pandas as pd
from IPython.display import display

from settings import EXPERIMENT_NAME, ExperimentPaths, get_nested, load_config
from typewell_continuity_selector_confidence_replacement_only_on_exp148 import (
    EXP145_TRAIN_ML_FEATURES,
    EXP191_CONTINUITY_OOF_PREDICTIONS,
    FULL_REPLAY_TRAIN_FEATURES,
    OUTPUT_PREFIX,
    find_artifact,
    run_typewell_continuity_selector_confidence_replacement_only_on_exp148,
)

LGB_CONFIG_INDEX = 2


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
booster_count = len(active_variants) * len(active_modes) * n_folds

print("Experiment:", EXPERIMENT_NAME)
print("Split train config:", f"lgb{LGB_CONFIG_INDEX}")
print("Route:", cfg_get(config, "experiment.route"))
print("Parent:", cfg_get(config, "lineage.parent"))
print("Continuity selector parent:", cfg_get(config, "lineage.continuity_selector_parent"))
print("Candidate ranker parent:", cfg_get(config, "lineage.candidate_ranker_parent"))
print("Kernel sources:", cfg_get(config, "runtime.kaggle.train_lgb_kernel_sources"))
print("Active modes:", active_modes)
print("Active variants:", [variant["name"] for variant in active_variants])
print("Planned boosters in this split:", booster_count)

# %% [markdown]
# ## 2. Input and continuity selector contract

# %%
cache_path = find_artifact(
    FULL_REPLAY_TRAIN_FEATURES,
    cfg_get(config, "data.exp072_train_feature_cache_local"),
)
learned_path = find_artifact(
    EXP145_TRAIN_ML_FEATURES,
    cfg_get(config, "data.learned_likelihood_train_features_local"),
)
continuity_path = find_artifact(
    EXP191_CONTINUITY_OOF_PREDICTIONS,
    cfg_get(config, "data.exp191_continuity_oof_predictions_local"),
)
print("exp072 full replay train cache:", cache_path)
print("exp145 learned likelihood feature inventory:", learned_path)
print("exp191 continuity OOF selected path:", continuity_path)
print("Replacement boundary: learned_likelihood_confidence is excluded from the active model")

base_preview = pd.read_csv(cache_path, nrows=5, dtype={"id": str, "well": str})
continuity_preview = pd.read_csv(continuity_path, nrows=1000, dtype={"id": str, "well": str})
continuity_preview = continuity_preview[
    continuity_preview["variant"].astype(str).eq(
        str(cfg_get(config, "model.typewell_continuity_selector_features.selected_variant"))
    )
    & continuity_preview["mode"].astype(str).eq(
        str(cfg_get(config, "model.typewell_continuity_selector_features.selected_mode"))
    )
].head(5)
display(base_preview.head())
display(
    continuity_preview[
        ["id", "well", "selected_candidate", "selected_candidate_index", "selected_tvt"]
    ]
)

# %% [markdown]
# ## 3. Train selected LGB config

# %%
summary = run_typewell_continuity_selector_confidence_replacement_only_on_exp148(
    output_dir=paths.artifacts_dir,
    train_dir=paths.train_data_dir,
    cache_path=cfg_get(config, "data.exp072_train_feature_cache_local"),
    learned_feature_path=cfg_get(config, "data.learned_likelihood_train_features_local"),
    learned_schema_path=cfg_get(config, "data.learned_likelihood_train_feature_schema_local"),
    learned_summary_path=cfg_get(config, "data.learned_likelihood_train_summary_local"),
    projection_config=cfg_get(config, "model.u_projection", {}),
    learned_feature_config=cfg_get(config, "model.learned_likelihood_features", {}),
    typewell_continuity_feature_config=cfg_get(
        config,
        "model.typewell_continuity_selector_features",
        {},
    ),
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
    top_n_importance=int(cfg_get(config, "model.training.top_n_importance", 60)),
    selected_lgb_config_indices=[LGB_CONFIG_INDEX],
)
paths.metrics_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
print(json.dumps(summary.get("best_lgb_mean_by_rmse_tvt"), indent=2, ensure_ascii=False))

# %% [markdown]
# ## 4. Metrics and generated artifacts

# %%
metrics = pd.read_csv(paths.artifacts_dir / f"{OUTPUT_PREFIX}_metrics.csv")
continuity_summary = pd.read_csv(
    paths.artifacts_dir / f"{OUTPUT_PREFIX}_exp191_continuity_feature_summary.csv"
)
importance_mean = pd.read_csv(paths.artifacts_dir / f"{OUTPUT_PREFIX}_feature_importance_mean.csv")
manifest_path = paths.artifacts_dir / f"{OUTPUT_PREFIX}_lgb_models" / "manifest.json"

display(metrics[metrics["fold"].astype(str).eq("pooled")].sort_values("rmse_tvt"))
display(continuity_summary)
display(importance_mean.head(60))
print("Model manifest:", manifest_path, "exists=", manifest_path.exists())
