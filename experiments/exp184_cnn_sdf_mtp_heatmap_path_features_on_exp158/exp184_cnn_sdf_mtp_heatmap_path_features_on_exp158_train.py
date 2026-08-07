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
# # exp184_cnn_sdf_mtp_heatmap_path_features_on_exp158 train
#
# Add-only exp182 CNN/SDF/MTP heatmap path confidence features for the exp157/158
# PF/Beam candidate selector. Heatmap path centers are used as selector features
# only; no direct replacement, softmax blend, PF-weight replacement, inference
# port, or submission candidate is created.

# %% [markdown]
# ## Contents
#
# 1. Imports
# 2. Runtime and configuration helpers
# 3. Input and feature contract checks
# 4. Train heatmap add-only selector scores and Viterbi continuity variants
# 5. Metrics, diagnostics, and generated artifacts

# %% [markdown]
# ## 1. Imports

# %%
from __future__ import annotations

from IPython.display import display

from cnn_sdf_mtp_heatmap_path_features_on_exp158 import (
    run_cnn_sdf_mtp_heatmap_path_features_on_exp158,
)
from settings import EXPERIMENT_NAME, ExperimentPaths, get_nested, load_config

# %% [markdown]
# ## 2. Runtime and configuration helpers

# %%
paths = ExperimentPaths()
config = load_config()
paths.require_kaggle_runtime()
paths.ensure_output_dirs()

heatmap_config = get_nested(config, "ranker.heatmap_path_features") or {}
selector_config = get_nested(config, "selector") or {}
runtime_config = get_nested(config, "runtime.kaggle") or {}
model_config = get_nested(config, "model") or {}

candidates = [item.get("name") for item in get_nested(config, "ranker.candidates") or []]
viterbi_grid = get_nested(config, "selector.viterbi_grid") or {}
viterbi_variant_count = (
    len(viterbi_grid.get("switch_penalty", []))
    * len(viterbi_grid.get("nondefault_bias", []))
    * len(viterbi_grid.get("jump_penalty_weight", []))
    * len(viterbi_grid.get("jump_free_ft", []))
    * len(viterbi_grid.get("max_abs_delta_vs_default", []))
    * len(viterbi_grid.get("max_pf_ancc_std", []))
    * len(viterbi_grid.get("min_md_since", []))
    * len(viterbi_grid.get("min_segment_len", []))
)

print("Experiment:", EXPERIMENT_NAME)
print("Route:", get_nested(config, "experiment.route"))
print("Mode:", get_nested(config, "audit.mode"))
print("Parent:", get_nested(config, "lineage.parent"))
print("Ranker parent:", get_nested(config, "lineage.ranker_parent"))
print("Heatmap parent:", get_nested(config, "lineage.heatmap_parent"))
print("Artifacts:", paths.artifacts_dir)
print("GPU enabled:", runtime_config.get("enable_gpu"))
print("Candidates:", candidates)
print("Heatmap primary run spec:", heatmap_config.get("primary_run_spec"))
print("Heatmap control run specs:", heatmap_config.get("shuffled_run_spec"), heatmap_config.get("no_gr_run_spec"))
print("Viterbi variants:", viterbi_variant_count)
print(
    "LightGBM configs:",
    model_config.get("planned_lightgbm_configs"),
    "folds:",
    model_config.get("planned_folds"),
    "boosters:",
    model_config.get("planned_boosters"),
    "control retraining:",
    model_config.get("control_or_parent_retraining"),
)

# %% [markdown]
# ## 3. Input and feature contract checks

# %%
display(
    {
        "data_sources": {
            "exp099": get_nested(config, "data.exp099_train_feature_cache_local"),
            "exp072": get_nested(config, "data.exp072_train_feature_cache_local"),
            "exp182_validation_predictions": get_nested(
                config, "data.exp182_validation_predictions_local"
            ),
            "exp182_sample_index": get_nested(config, "data.exp182_sample_index_local"),
            "exp182_summary": get_nested(config, "data.exp182_summary_local"),
            "exp115": get_nested(config, "data.exp115_fold_assignments_local"),
        },
        "heatmap_feature_policy": {
            "interpolation": heatmap_config.get("interpolation"),
            "topk_ranks": heatmap_config.get("topk_ranks"),
            "excluded_run_specs": heatmap_config.get("excluded_run_specs"),
            "forbidden_sources": [
                "pred_top*_abs_error",
                "top*_within10",
                "true_center_tvt",
                "oracle labels",
            ],
        },
        "historical_baselines": selector_config.get("historical_baselines"),
        "expected_train_artifacts": get_nested(config, "audit.expected_train_artifacts"),
    }
)

# %% [markdown]
# ## 4. Train heatmap add-only selector scores and Viterbi continuity variants

# %%
summary = run_cnn_sdf_mtp_heatmap_path_features_on_exp158(
    output_dir=paths.artifacts_dir,
    cache_path=get_nested(config, "data.exp099_train_feature_cache_local"),
    schema_path=get_nested(config, "data.exp099_train_feature_schema_local"),
    max_rows=get_nested(config, "ranker.max_rows"),
)

display(
    {
        "status": summary["status"],
        "runtime_seconds": summary["runtime_seconds"],
        "rows": summary["rows"],
        "wells": summary["wells"],
        "feature_count": summary["feature_count"],
        "heatmap_feature_count": summary["heatmap_path_features"]["generated_feature_count"],
        "best_metric": summary["best_metric"],
        "best_viterbi_variant": summary["best_viterbi_variant"],
        "decision": summary["decision"],
    }
)

# %% [markdown]
# ## 5. Metrics, diagnostics, and generated artifacts

# %%
import pandas as pd

metrics = pd.read_csv(paths.artifacts_dir / summary["artifacts"]["metrics"])
distribution = pd.read_csv(paths.artifacts_dir / summary["artifacts"]["selection_distribution"])
by_well = pd.read_csv(paths.artifacts_dir / summary["artifacts"]["by_well"])
bucket_metrics = pd.read_csv(paths.artifacts_dir / summary["artifacts"]["bucket_metrics"])
subgroup_metrics = pd.read_csv(paths.artifacts_dir / summary["artifacts"]["subgroup_metrics"])
importance = pd.read_csv(paths.artifacts_dir / summary["artifacts"]["feature_importance_mean"])
score_summary = pd.read_csv(paths.artifacts_dir / summary["artifacts"]["score_summary"])
heatmap_summary = pd.read_csv(paths.artifacts_dir / summary["artifacts"]["heatmap_feature_summary"])

print("overall metrics")
display(metrics.head(80))

print("selection distribution")
display(distribution.head(120))

print("worst wells")
display(
    by_well.sort_values(
        ["variant", "mode", "rmse_tvt"],
        ascending=[True, True, False],
    ).head(120)
)

print("distance / heatmap / stress bucket metrics")
display(bucket_metrics.head(200))

print("subgroup metrics")
display(subgroup_metrics.head(200))

print("feature importance mean")
display(importance.head(140))

print("score summary")
display(score_summary.head(80))

print("heatmap feature summary")
display(heatmap_summary.head(120))

print("generated artifacts")
for key, value in summary["artifacts"].items():
    print(f"{key}: {value}")
