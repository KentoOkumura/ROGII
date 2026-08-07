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
# # exp199_typewell_hard_window_pct40_base_surface_keep_exp145_ll_on_exp148 train
#
# Full-train mixed-provenance diagnostic that replaces the exp148 base PF/Beam
# surface with exp196 pct40 hard-window cache while keeping exp145 `ll_*`
# learned-likelihood confidence features unchanged.
#

# %% [markdown]
# ## Contents
#
# 1. Setup and configuration
# 2. Input and full-train coverage contract
# 3. Train pct40-base keep-ll diagnostic variant
# 4. Metrics and generated artifacts
#

# %% [markdown]
# ## 1. Setup and configuration

# %%
from __future__ import annotations

import json

import pandas as pd
from IPython.display import display

from settings import EXPERIMENT_NAME, ExperimentPaths, get_nested, load_config
from typewell_hard_window_pct40_base_surface_keep_exp145_ll_on_exp148 import (
    EXP145_TRAIN_ML_FEATURES,
    FULL_REPLAY_TRAIN_FEATURES,
    OUTPUT_PREFIX,
    find_artifact,
    load_learned_likelihood_ml_features,
    run_typewell_hard_window_pct40_base_surface_keep_exp145_ll_on_exp148,
)

def cfg_get(config, dotted_key, default=None):
    value = get_nested(config, dotted_key)
    return default if value is None else value

paths = ExperimentPaths()
paths.require_kaggle_runtime()
paths.ensure_output_dirs()
config = load_config()

active_variants = [v for v in cfg_get(config, "model.feature_ablation.active_variants", []) if v.get("enabled", True)]
active_modes = cfg_get(config, "model.training.active_modes", [])
lgb_config_count = 3
n_folds = int(cfg_get(config, "validation.n_folds", 5))
booster_count = len(active_variants) * len(active_modes) * lgb_config_count * n_folds

print("Experiment:", EXPERIMENT_NAME)
print("Route:", cfg_get(config, "experiment.route"))
print("Mode:", cfg_get(config, "audit.mode"))
print("Parent:", cfg_get(config, "lineage.parent"))
print("Base surface parent:", cfg_get(config, "lineage.base_surface_parent"))
print("Learned likelihood parent:", cfg_get(config, "lineage.learned_likelihood_parent"))
print("Kernel sources:", cfg_get(config, "runtime.kaggle.kernel_sources"))
print("Active modes:", active_modes)
print("Active variants:", [v["name"] for v in active_variants])
print("Active feature groups:", {v["name"]: v.get("feature_groups") for v in active_variants})
print("Planned LightGBM configs:", lgb_config_count, "folds:", n_folds, "boosters:", booster_count)


# %% [markdown]
# ## 2. Input and full-train coverage contract

# %%
cache_path = find_artifact(
    FULL_REPLAY_TRAIN_FEATURES,
    cfg_get(config, "data.base_surface_train_feature_cache_local"),
)
learned_path = find_artifact(
    EXP145_TRAIN_ML_FEATURES,
    cfg_get(config, "data.learned_likelihood_train_features_local"),
)
print("exp196 pct40 hard-window base surface cache:", cache_path)
print("exp145 full-train learned likelihood feature cache:", learned_path)
print("Mixed provenance:", "base/projection from exp196 pct40; ll_* from exp145")

base_preview = pd.read_csv(cache_path, nrows=5, dtype={"id": str, "well": str})
learned_preview, learned_meta = load_learned_likelihood_ml_features(
    cfg_get(config, "data.learned_likelihood_train_features_local"),
    schema_path=cfg_get(config, "data.learned_likelihood_train_feature_schema_local"),
    summary_path=cfg_get(config, "data.learned_likelihood_train_summary_local"),
)
print("learned feature rows:", learned_meta["rows"], "wells:", learned_meta["wells"], "columns:", learned_meta["columns"])
preview_cols = [
    c
    for c in [
        "id",
        "well",
        "target",
        "last_known_tvt",
        "z",
        "md_since",
        "pf_ancc",
        "likpf_mean_d",
    ]
    if c in base_preview.columns
]
display(base_preview[preview_cols])
display(learned_preview.head()[[
    "id",
    "well",
    "fold",
    "md_since",
    "learned_prob_top1_value",
    "learned_prob_entropy",
    "learned_pred_abs_error_likpf_mean",
    "candidate_tvt_likpf_mean",
]])


# %% [markdown]
# ## 3. Train pct40-base keep-ll diagnostic variant

# %%
summary = run_typewell_hard_window_pct40_base_surface_keep_exp145_ll_on_exp148(
    output_dir=paths.artifacts_dir,
    train_dir=paths.train_data_dir,
    cache_path=cfg_get(config, "data.base_surface_train_feature_cache_local"),
    learned_feature_path=cfg_get(config, "data.learned_likelihood_train_features_local"),
    learned_schema_path=cfg_get(config, "data.learned_likelihood_train_feature_schema_local"),
    learned_summary_path=cfg_get(config, "data.learned_likelihood_train_summary_local"),
    projection_config=cfg_get(config, "model.u_projection", {}),
    learned_feature_config=cfg_get(config, "model.learned_likelihood_features", {}),
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
)
print(json.dumps({
    "status": summary["status"],
    "best_lgb_mean_by_rmse_tvt": summary["best_lgb_mean_by_rmse_tvt"],
    "feature_join_coverage": summary["feature_join_coverage"],
}, indent=2))


# %% [markdown]
# ## 4. Metrics and generated artifacts

# %%
metrics = pd.read_csv(paths.artifacts_dir / f"{OUTPUT_PREFIX}_metrics.csv")
by_well = pd.read_csv(paths.artifacts_dir / f"{OUTPUT_PREFIX}_by_well.csv")
bucket_metrics = pd.read_csv(paths.artifacts_dir / f"{OUTPUT_PREFIX}_bucket_metrics.csv")
projection_summary = pd.read_csv(paths.artifacts_dir / f"{OUTPUT_PREFIX}_projection_feature_summary.csv")
learned_summary = pd.read_csv(paths.artifacts_dir / f"{OUTPUT_PREFIX}_learned_feature_summary.csv")
importance_mean = pd.read_csv(paths.artifacts_dir / f"{OUTPUT_PREFIX}_feature_importance_mean.csv")
manifest_path = paths.artifacts_dir / f"{OUTPUT_PREFIX}_lgb_models" / "manifest.json"

pooled = metrics[metrics["fold"].astype(str).eq("pooled")].sort_values("rmse_tvt")
display(pooled)
display(learned_summary)
display(projection_summary)
display(bucket_metrics.head(50))
display(by_well.head(30))
display(importance_mean.head(60))
print("Model manifest:", manifest_path, "exists=", manifest_path.exists())
print("Feature importance plot:", paths.artifacts_dir / f"{OUTPUT_PREFIX}_feature_importance_mean_top.png")
