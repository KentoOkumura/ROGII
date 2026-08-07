# %% [markdown]
# # exp164_spatial_prior_confidence_features_on_exp092_kaggle train_lgb1
#
# CPU train notebook for only the `lgb1` config.

# %% [markdown]
# ## Contents
#
# 1. Imports
# 2. Runtime and configuration helpers
# 3. Input and spatial prior feature contract
# 4. Training orchestration
# 5. Metrics and generated artifacts

# %% [markdown]
# ## 1. Imports

# %%
from __future__ import annotations

import json

import pandas as pd
from IPython.display import display
from settings import EXPERIMENT_NAME, ExperimentPaths, get_nested, load_config
from spatial_prior_confidence_features_on_exp092_kaggle import (
    EXP114_ARTIFACTS,
    EXP114_SPATIAL_OOF,
    FULL_REPLAY_TRAIN_FEATURES,
    OUTPUT_PREFIX,
    find_artifact,
    run_spatial_prior_confidence_features_on_exp092,
)

# %% [markdown]
# ## 2. Runtime and configuration helpers

# %%
SELECTED_LGB_CONFIGS = ["lgb1"]


def cfg_get(config, dotted_key, default=None):
    value = get_nested(config, dotted_key)
    return default if value is None else value


def enabled_variant_names(config):
    variants = cfg_get(config, "model.feature_ablation.active_variants", [])
    return [variant["name"] for variant in variants if variant.get("enabled", True)]


paths = ExperimentPaths()
paths.require_kaggle_runtime()
paths.ensure_output_dirs()
config = load_config()

active_modes = cfg_get(config, "model.training.active_modes", [])
active_variants = enabled_variant_names(config)
n_folds = int(cfg_get(config, "validation.n_folds", 5))

print("Experiment:", EXPERIMENT_NAME)
print("Notebook:", "train_lgb1")
print("Route:", cfg_get(config, "experiment.route"))
print("Mode:", cfg_get(config, "audit.mode"))
print("Parent:", cfg_get(config, "lineage.parent"))
print("Spatial prior parent:", cfg_get(config, "lineage.spatial_prior_parent"))
print("Kernel sources:", cfg_get(config, "runtime.kaggle.kernel_sources"))
print("Active modes:", active_modes)
print("Active variants:", active_variants)
print("Selected LGB configs:", SELECTED_LGB_CONFIGS)
print(
    "Planned boosters:",
    len(active_variants) * len(active_modes) * len(SELECTED_LGB_CONFIGS) * n_folds,
)

# %% [markdown]
# ## 3. Input and spatial prior feature contract

# %%
cache_path = find_artifact(
    FULL_REPLAY_TRAIN_FEATURES,
    cfg_get(config, "data.exp072_train_feature_cache_local"),
)
spatial_path = find_artifact(
    EXP114_SPATIAL_OOF,
    cfg_get(config, "data.exp114_spatial_oof_local"),
    local_artifacts=EXP114_ARTIFACTS,
)

print("exp072 full replay train cache:", cache_path)
print("exp114 spatial prior OOF:", spatial_path)

base_preview = pd.read_csv(cache_path, nrows=5, dtype={"id": str, "well": str})
spatial_preview = pd.read_csv(spatial_path, nrows=5, dtype={"id": str, "well": str})
base_cols = [
    column
    for column in [
        "id",
        "well",
        "target",
        "last_known_tvt",
        "z",
        "md_since",
        "pf_ancc",
        "beam_mean_d",
        "likpf_mean_d",
    ]
    if column in base_preview.columns
]
spatial_cols = [
    column
    for column in [
        "id",
        "well",
        "xy_only_k8_prior_tvt",
        "xy_only_k8_prior_std",
        "xy_only_k8_neighbor_wells",
        "xy_only_k8_distance_mean",
        "xy_plus_trajectory_shape_k8_prior_tvt",
        "xy_plus_trajectory_shape_k8_prior_std",
    ]
    if column in spatial_preview.columns
]

display(base_preview[base_cols])
display(spatial_preview[spatial_cols])
print(
    "Spatial prior variants:",
    cfg_get(config, "model.spatial_prior_confidence.prior_variants", []),
)
print("Exp118 gate proxy:", cfg_get(config, "model.spatial_prior_confidence.exp118_best_gate", {}))

# %% [markdown]
# ## 4. Training orchestration

# %%
summary = run_spatial_prior_confidence_features_on_exp092(
    output_dir=paths.artifacts_dir,
    train_dir=paths.train_data_dir,
    cache_path=cfg_get(config, "data.exp072_train_feature_cache_local"),
    projection_config=cfg_get(config, "model.u_projection", {}),
    spatial_config=cfg_get(config, "model.spatial_prior_confidence", {}),
    variants=cfg_get(config, "model.feature_ablation.active_variants", []),
    modes=cfg_get(config, "model.training.modes", {}),
    active_modes=active_modes,
    n_splits=n_folds,
    fast=bool(cfg_get(config, "audit.fast", False)),
    early_stopping_rounds=int(cfg_get(config, "model.training.early_stopping_rounds", 250)),
    max_rows=cfg_get(config, "model.training.max_rows"),
    max_train_rows=cfg_get(config, "model.training.max_train_rows"),
    save_models=bool(cfg_get(config, "model.training.save_models", True)),
    save_predictions=bool(cfg_get(config, "model.training.save_predictions", True)),
    top_n_importance=int(cfg_get(config, "model.training.top_n_importance", 60)),
    selected_lgb_configs=SELECTED_LGB_CONFIGS,
)
paths.metrics_path.write_text(
    json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
)
print(
    json.dumps(
        {
            "status": summary["status"],
            "selected_lgb_configs": summary["selected_lgb_configs"],
            "best_lgb_mean_by_rmse_tvt": summary["best_lgb_mean_by_rmse_tvt"],
        },
        indent=2,
        ensure_ascii=False,
    )[:4000]
)
print("Metrics written:", paths.metrics_path)

# %% [markdown]
# ## 5. Metrics and generated artifacts

# %%
metrics = pd.read_csv(paths.artifacts_dir / f"{OUTPUT_PREFIX}_metrics.csv")
by_well = pd.read_csv(paths.artifacts_dir / f"{OUTPUT_PREFIX}_by_well.csv")
bucket_metrics = pd.read_csv(paths.artifacts_dir / f"{OUTPUT_PREFIX}_bucket_metrics.csv")
projection_summary = pd.read_csv(
    paths.artifacts_dir / f"{OUTPUT_PREFIX}_projection_feature_summary.csv"
)
spatial_summary = pd.read_csv(paths.artifacts_dir / f"{OUTPUT_PREFIX}_spatial_feature_summary.csv")
importance_mean = pd.read_csv(paths.artifacts_dir / f"{OUTPUT_PREFIX}_feature_importance_mean.csv")
manifest_path = paths.artifacts_dir / f"{OUTPUT_PREFIX}_lgb_models" / "manifest.json"

pooled = metrics[metrics["fold"].astype(str).eq("pooled")].sort_values("rmse_tvt")
display(pooled)
display(spatial_summary)
display(projection_summary)
display(bucket_metrics.head(50))
display(by_well.head(30))
display(importance_mean.head(60))
print("Model manifest:", manifest_path, "exists=", manifest_path.exists())
print(
    "Feature importance plot:",
    paths.artifacts_dir / f"{OUTPUT_PREFIX}_feature_importance_mean_top.png",
)
