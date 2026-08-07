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
# # exp220 row-neighbor context train lgb2
#
# CPU split training for LightGBM config `lgb2` only.

# %% [markdown]
# ## Contents
#
# 1. Setup and configuration
# 2. Input and feature contract
# 3. Train row-neighbor add-only variant
# 4. Metrics and generated artifacts

# %% [markdown]
# ## 1. Setup and configuration

# %%
from __future__ import annotations

import gc
import json

import pandas as pd
from IPython.display import display

from row_neighbor_input_context_features_on_exp148 import (
    EXP145_TRAIN_ML_FEATURES,
    FULL_REPLAY_TRAIN_FEATURES,
    OUTPUT_PREFIX,
    find_artifact,
    run_row_neighbor_input_context_features_on_exp148,
)
from settings import EXPERIMENT_NAME, ExperimentPaths, get_nested, load_config


LGB_CONFIG_INDEX = 2


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
booster_count = len(active_variants) * len(active_modes) * n_folds

print("Experiment:", EXPERIMENT_NAME)
print("Route:", cfg_get(config, "experiment.route"))
print("Mode:", cfg_get(config, "audit.mode"))
print("Parent:", cfg_get(config, "lineage.parent"))
print("Selected LightGBM config:", f"lgb{LGB_CONFIG_INDEX}")
print("Active modes:", active_modes)
print("Active variants:", [v["name"] for v in active_variants])
print("Planned folds:", n_folds, "boosters in this split:", booster_count)

# %% [markdown]
# ## 2. Input and feature contract

# %%
cache_path = find_artifact(
    FULL_REPLAY_TRAIN_FEATURES,
    cfg_get(config, "data.exp072_train_feature_cache_local"),
)
learned_path = find_artifact(
    EXP145_TRAIN_ML_FEATURES,
    cfg_get(config, "data.learned_likelihood_train_features_local"),
)
print("exp072 full replay train cache:", cache_path)
print("exp145 full-train learned likelihood feature cache:", learned_path)
print(
    "Row-neighbor context feature config:",
    cfg_get(config, "model.row_neighbor_input_context_features"),
)

base_preview = pd.read_csv(cache_path, nrows=5, dtype={"id": str, "well": str})
learned_preview = pd.read_csv(learned_path, nrows=5, dtype={"id": str, "well": str})
preview_cols = [
    c
    for c in [
        "id",
        "well",
        "target",
        "last_known_tvt",
        "z",
        "md_since",
        "gr",
        "dzdmd",
        "beam_mean_d",
        "likpf_mean_d",
    ]
    if c in base_preview.columns
]
display(base_preview[preview_cols])
display(
    learned_preview.head()[
        [
            c
            for c in [
                "id",
                "well",
                "fold",
                "md_since",
                "learned_prob_entropy",
                "candidate_tvt_std",
                "candidate_tvt_range",
            ]
            if c in learned_preview.columns
        ]
    ]
)
del base_preview, learned_preview
gc.collect()

# %% [markdown]
# ## 3. Train row-neighbor add-only variant

# %%
summary = run_row_neighbor_input_context_features_on_exp148(
    output_dir=paths.artifacts_dir,
    train_dir=paths.train_data_dir,
    cache_path=cfg_get(config, "data.exp072_train_feature_cache_local"),
    learned_feature_path=cfg_get(config, "data.learned_likelihood_train_features_local"),
    learned_schema_path=cfg_get(config, "data.learned_likelihood_train_feature_schema_local"),
    learned_summary_path=cfg_get(config, "data.learned_likelihood_train_summary_local"),
    projection_config=cfg_get(config, "model.u_projection", {}),
    learned_feature_config=cfg_get(config, "model.learned_likelihood_features", {}),
    row_context_feature_config=cfg_get(
        config,
        "model.row_neighbor_input_context_features",
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
    lgb_config_indices=[LGB_CONFIG_INDEX],
)
print(
    json.dumps(
        {
            "status": summary["status"],
            "pooled_metrics": summary["pooled_metrics"],
            "feature_join_coverage": summary["feature_join_coverage"],
        },
        indent=2,
    )
)

# %% [markdown]
# ## 4. Metrics and generated artifacts

# %%
metrics = pd.read_csv(paths.artifacts_dir / f"{OUTPUT_PREFIX}_metrics.csv")
by_well = pd.read_csv(paths.artifacts_dir / f"{OUTPUT_PREFIX}_by_well.csv")
bucket_metrics = pd.read_csv(paths.artifacts_dir / f"{OUTPUT_PREFIX}_bucket_metrics.csv")
row_context_summary = pd.read_csv(
    paths.artifacts_dir / f"{OUTPUT_PREFIX}_row_context_feature_summary.csv"
)
importance_mean = pd.read_csv(paths.artifacts_dir / f"{OUTPUT_PREFIX}_feature_importance_mean.csv")
manifest_path = paths.artifacts_dir / f"{OUTPUT_PREFIX}_lgb_models" / "manifest.json"

display(metrics[metrics["fold"].astype(str).eq("pooled")].sort_values("rmse_tvt"))
display(row_context_summary.head(60))
display(bucket_metrics.head(50))
display(by_well.head(30))
display(importance_mean.head(60))
print("Model manifest:", manifest_path, "exists=", manifest_path.exists())
print(
    "Feature importance plot:",
    paths.artifacts_dir / f"{OUTPUT_PREFIX}_feature_importance_mean_top.png",
)
