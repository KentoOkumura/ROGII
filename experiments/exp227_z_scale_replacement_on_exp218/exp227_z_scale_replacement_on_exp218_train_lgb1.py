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
# # exp227_z_scale_replacement_on_exp218 train lgb1
#
# CPU split training for LightGBM config `lgb1` only.

# %% [markdown]
# ## Contents
#
# 1. Setup and configuration
# 2. Input and full-train coverage contract
# 3. Train selected LightGBM config
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
from z_scale_replacement_on_exp218 import (
    EXP145_TRAIN_ML_FEATURES,
    FULL_REPLAY_TRAIN_FEATURES,
    OUTPUT_PREFIX,
    find_artifact,
    run_z_scale_replacement_on_exp218,
)

LGB_CONFIG_INDEX = 1


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
print("Route:", cfg_get(config, "experiment.route"))
print("Mode:", cfg_get(config, "audit.mode"))
print("Parent:", cfg_get(config, "lineage.parent"))
print("ML submitted anchor:", cfg_get(config, "lineage.ml_submitted_anchor_parent"))
print("Selected LightGBM config:", f"lgb{LGB_CONFIG_INDEX}")
print("Kernel sources:", cfg_get(config, "runtime.kaggle.train_kernel_sources"))
print("Active modes:", active_modes)
print("Active variants:", [variant["name"] for variant in active_variants])
print("Dropped base columns:", {variant["name"]: variant.get("drop_base_columns", []) for variant in active_variants})
print("Planned folds:", n_folds, "boosters in this split:", booster_count)

# %% [markdown]
# ## 2. Input and full-train coverage contract

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
print("GRWR feature config:", cfg_get(config, "model.gr_wavelet_rotation_confidence_features"))
print("Well-scaled z/dz feature config:", cfg_get(config, "model.well_scaled_z_dz_features"))

base_preview = pd.read_csv(cache_path, nrows=5, dtype={"id": str, "well": str})
learned_preview = pd.read_csv(learned_path, nrows=5, dtype={"id": str, "well": str})
print("learned feature preview rows:", len(learned_preview), "columns:", len(learned_preview.columns))
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
display(
    learned_preview.head()[
        [
            "id",
            "well",
            "fold",
            "md_since",
            "learned_prob_top1_value",
            "learned_prob_entropy",
            "learned_pred_abs_error_likpf_mean",
            "candidate_tvt_likpf_mean",
        ]
    ]
)
train_files = sorted(paths.train_data_dir.glob("*__horizontal_well.csv"))[:3]
print("raw train horizontal preview files:", [path.name for path in train_files])
del base_preview, learned_preview
gc.collect()

# %% [markdown]
# ## 3. Train selected LightGBM config

# %%
summary = run_z_scale_replacement_on_exp218(
    output_dir=paths.artifacts_dir,
    train_dir=paths.train_data_dir,
    cache_path=cfg_get(config, "data.exp072_train_feature_cache_local"),
    learned_feature_path=cfg_get(config, "data.learned_likelihood_train_features_local"),
    learned_schema_path=cfg_get(config, "data.learned_likelihood_train_feature_schema_local"),
    learned_summary_path=cfg_get(config, "data.learned_likelihood_train_summary_local"),
    projection_config=cfg_get(config, "model.u_projection", {}),
    learned_feature_config=cfg_get(config, "model.learned_likelihood_features", {}),
    grwr_feature_config=cfg_get(config, "model.gr_wavelet_rotation_confidence_features", {}),
    well_scaled_feature_config=cfg_get(config, "model.well_scaled_z_dz_features", {}),
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
            "selected_lgb_config_indices": summary["lgb_config_indices"],
            "best_lgb_mean_by_rmse_tvt": summary["best_lgb_mean_by_rmse_tvt"],
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
projection_summary = pd.read_csv(paths.artifacts_dir / f"{OUTPUT_PREFIX}_projection_feature_summary.csv")
learned_summary = pd.read_csv(paths.artifacts_dir / f"{OUTPUT_PREFIX}_learned_feature_summary.csv")
grwr_summary = pd.read_csv(paths.artifacts_dir / f"{OUTPUT_PREFIX}_grwr_feature_summary.csv")
well_scaled_summary = pd.read_csv(
    paths.artifacts_dir / f"{OUTPUT_PREFIX}_well_scaled_z_dz_feature_summary.csv"
)
importance_mean = pd.read_csv(paths.artifacts_dir / f"{OUTPUT_PREFIX}_feature_importance_mean.csv")
manifest_path = paths.artifacts_dir / f"{OUTPUT_PREFIX}_lgb_models" / "manifest.json"

display(metrics.sort_values(["variant", "mode", "model", "fold"]).tail(12))
display(bucket_metrics.sort_values("rmse_tvt", ascending=False).head(12))
display(by_well.head(12))
display(projection_summary.head(12))
display(learned_summary.head(12))
display(grwr_summary.head(12))
display(well_scaled_summary.head(20))
display(importance_mean.head(40))
print("Manifest:", manifest_path, "exists=", manifest_path.exists())
print("Artifacts dir:", paths.artifacts_dir)
