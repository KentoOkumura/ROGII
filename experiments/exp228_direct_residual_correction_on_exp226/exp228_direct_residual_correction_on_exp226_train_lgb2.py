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
# # exp228_direct_residual_correction_on_exp226 train_lgb2
#
# CPU split training for the `lgb2` LightGBM config. The feature surface matches exp218; the target is `TVT - exp226_oof_pred`.

# %% [markdown]
# ## Contents
#
# 1. Setup and configuration
# 2. Input and residual target contract
# 3. Train selected split
# 4. Metrics and generated artifacts

# %% [markdown]
# ## 1. Setup and configuration

# %%
from __future__ import annotations

import gc
import json

import pandas as pd
from IPython.display import display

from settings import EXPERIMENT_NAME, ExperimentPaths, get_nested, load_config
from direct_residual_correction_on_exp226 import (
    EXP145_TRAIN_ML_FEATURES,
    EXP226_TRAIN_OOF_PREDICTIONS,
    FULL_REPLAY_TRAIN_FEATURES,
    OUTPUT_PREFIX,
    find_artifact,
    run_direct_residual_correction_on_exp226,
)

SPLIT_KIND = "train_lgb2"
SELECTED_LGB_MODELS = ["lgb2"]


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
booster_count = len(active_variants) * len(active_modes) * len(SELECTED_LGB_MODELS) * n_folds

print("Experiment:", EXPERIMENT_NAME)
print("Split:", SPLIT_KIND)
print("Route:", cfg_get(config, "experiment.route"))
print("Parent:", cfg_get(config, "lineage.parent"))
print("ML feature surface parent:", cfg_get(config, "lineage.ml_feature_surface_parent"))
print("Active modes:", active_modes)
print("Active variants:", [variant["name"] for variant in active_variants])
print("Selected LGB models:", SELECTED_LGB_MODELS)
print("Planned folds:", n_folds, "boosters:", booster_count)
print("Kaggle GPU enabled:", cfg_get(config, "runtime.kaggle.enable_gpu"))

# %% [markdown]
# ## 2. Input and residual target contract

# %%
cache_path = find_artifact(
    FULL_REPLAY_TRAIN_FEATURES,
    cfg_get(config, "data.exp072_train_feature_cache_local"),
)
learned_path = find_artifact(
    EXP145_TRAIN_ML_FEATURES,
    cfg_get(config, "data.learned_likelihood_train_features_local"),
)
exp226_oof_path = find_artifact(
    EXP226_TRAIN_OOF_PREDICTIONS,
    cfg_get(config, "data.exp226_train_oof_local"),
)
print("exp072 full replay train cache:", cache_path)
print("exp145 full-train learned likelihood feature cache:", learned_path)
print("exp226 group-safe OOF predictions:", exp226_oof_path)
print("GRWR feature config:", cfg_get(config, "model.gr_wavelet_rotation_confidence_features"))

base_preview = pd.read_csv(cache_path, nrows=5, dtype={"id": str, "well": str})
exp226_preview = pd.read_csv(exp226_oof_path, nrows=5, dtype={"well_id": str})
preview_cols = [
    column
    for column in [
        "id",
        "well",
        "target",
        "last_known_tvt",
        "z",
        "md_since",
        "pf_ancc",
        "likpf_mean_d",
    ]
    if column in base_preview.columns
]
display(base_preview[preview_cols])
display(exp226_preview)
train_files = sorted(paths.train_data_dir.glob("*__horizontal_well.csv"))[:3]
print("raw train horizontal preview files:", [path.name for path in train_files])
del base_preview, exp226_preview
gc.collect()

# %% [markdown]
# ## 3. Train selected split

# %%
summary = run_direct_residual_correction_on_exp226(
    output_dir=paths.artifacts_dir,
    train_dir=paths.train_data_dir,
    cache_path=cfg_get(config, "data.exp072_train_feature_cache_local"),
    exp226_oof_path=cfg_get(config, "data.exp226_train_oof_local"),
    learned_feature_path=cfg_get(config, "data.learned_likelihood_train_features_local"),
    learned_schema_path=cfg_get(config, "data.learned_likelihood_train_feature_schema_local"),
    learned_summary_path=cfg_get(config, "data.learned_likelihood_train_summary_local"),
    projection_config=cfg_get(config, "model.u_projection", {}),
    learned_feature_config=cfg_get(config, "model.learned_likelihood_features", {}),
    grwr_feature_config=cfg_get(config, "model.gr_wavelet_rotation_confidence_features", {}),
    variants=cfg_get(config, "model.feature_ablation.active_variants", []),
    modes=cfg_get(config, "model.training.modes", {}),
    active_modes=cfg_get(config, "model.training.active_modes", []),
    n_splits=n_folds,
    fast=bool(cfg_get(config, "audit.fast", False)),
    early_stopping_rounds=int(cfg_get(config, "model.training.early_stopping_rounds", 250)),
    max_rows=cfg_get(config, "model.training.max_rows"),
    max_train_rows=cfg_get(config, "model.training.max_train_rows"),
    save_models=bool(cfg_get(config, "model.training.save_models", True)),
    save_predictions=bool(cfg_get(config, "model.training.save_predictions", True)),
    top_n_importance=int(cfg_get(config, "model.training.top_n_importance", 60)),
    selected_lgb_models=SELECTED_LGB_MODELS,
)
print(json.dumps({
    "status": summary["status"],
    "selected_lgb_models": summary["selected_lgb_models"],
    "best_lgb_mean_by_rmse_tvt": summary["best_lgb_mean_by_rmse_tvt"],
    "feature_join_coverage": summary["feature_join_coverage"],
    "exp226_oof_source": summary["exp226_oof_source"],
}, indent=2))

# %% [markdown]
# ## 4. Metrics and generated artifacts

# %%
metrics = pd.read_csv(paths.artifacts_dir / f"{OUTPUT_PREFIX}_metrics.csv")
by_well = pd.read_csv(paths.artifacts_dir / f"{OUTPUT_PREFIX}_by_well.csv")
bucket_metrics = pd.read_csv(paths.artifacts_dir / f"{OUTPUT_PREFIX}_bucket_metrics.csv")
importance_mean = pd.read_csv(paths.artifacts_dir / f"{OUTPUT_PREFIX}_feature_importance_mean.csv")
manifest_path = paths.artifacts_dir / f"{OUTPUT_PREFIX}_lgb_models" / "manifest.json"

pooled = metrics[metrics["fold"].astype(str).eq("pooled")].sort_values("rmse_tvt")
display(pooled)
display(bucket_metrics.head(50))
display(by_well.head(30))
display(importance_mean.head(60))
print("Model manifest:", manifest_path, "exists=", manifest_path.exists())
print("Feature importance plot:", paths.artifacts_dir / f"{OUTPUT_PREFIX}_feature_importance_mean_top.png")
