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
# # exp237_hmm_exp226_candidate_selector_on_exp183 train
#
# Add exp209 exact-HMM/likPF blend, exp223 self-GR HMM, and exp226 K16 geometry
# OOF paths to the exp183 selector. Train only the candidate absolute-error ranker
# and apply one fixed well-local continuity rule.

# %% [markdown]
# ## Contents
#
# 1. Imports
# 2. Runtime and configuration helpers
# 3. Input, OOF, and candidate contract checks
# 4. Train candidate-error scores and fixed Viterbi continuity selection
# 5. Metrics, diagnostics, and generated artifacts

# %% [markdown]
# ## 1. Imports

# %%
from __future__ import annotations

import pandas as pd
from hmm_exp226_candidate_selector_on_exp183 import (
    run_hmm_exp226_candidate_selector_on_exp183,
)
from IPython.display import display
from settings import EXPERIMENT_NAME, ExperimentPaths, get_nested, load_config

# %% [markdown]
# ## 2. Runtime and configuration helpers

# %%
paths = ExperimentPaths()
config = load_config()
paths.require_kaggle_runtime()
paths.ensure_output_dirs()

ranker_config = get_nested(config, "ranker") or {}
cluster_config = get_nested(config, "ranker.cluster_prior_features") or {}
selector_config = get_nested(config, "selector") or {}
runtime_config = get_nested(config, "runtime.kaggle") or {}
model_config = get_nested(config, "model") or {}

candidates = [item.get("name") for item in get_nested(config, "ranker.candidates") or []]
prior_variants = [
    item.get("name")
    for item in get_nested(config, "ranker.cluster_prior_features.prior_variants") or []
]
cluster_gates = [
    item.get("name")
    for item in get_nested(config, "ranker.cluster_prior_features.cluster_gates") or []
]
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
print("HMM parent:", get_nested(config, "lineage.hmm_parent"))
print("Self-GR HMM parent:", get_nested(config, "lineage.self_gr_hmm_parent"))
print("Geometry parent:", get_nested(config, "lineage.geometry_parent"))
print("Artifacts:", paths.artifacts_dir)
print("GPU enabled:", runtime_config.get("enable_gpu"))
print("Candidates:", candidates)
print("Prior variants:", prior_variants)
print("Cluster gates:", cluster_gates)
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
# ## 3. Input, OOF, and candidate contract checks

# %%
display(
    {
        "data_sources": {
            "exp099": get_nested(config, "data.exp099_train_feature_cache_local"),
            "exp072": get_nested(config, "data.exp072_train_feature_cache_local"),
            "exp109": get_nested(config, "data.exp109_oof_predictions_local"),
            "exp114": get_nested(config, "data.exp114_oof_predictions_local"),
            "exp065": get_nested(config, "data.exp065_cluster_assignments_local"),
            "exp115": get_nested(config, "data.exp115_fold_assignments_local"),
            "exp209_hmm": get_nested(config, "data.exp209_hmm_train_features_local"),
            "exp223_selfgr_hmm": get_nested(config, "data.exp223_train_features_local"),
            "exp226_k16": get_nested(config, "data.exp226_train_oof_local"),
        },
        "cluster_assignment_method": get_nested(config, "cluster.assignment_method"),
        "cluster_assignment_threshold": get_nested(config, "cluster.assignment_threshold"),
        "feature_correction_alpha": cluster_config.get("feature_correction_alpha"),
        "feature_correction_clips": cluster_config.get("feature_correction_clips"),
        "feature_quality_max_prior_std": cluster_config.get("feature_quality_max_prior_std"),
        "historical_baselines": selector_config.get("historical_baselines"),
        "expected_train_artifacts": get_nested(config, "audit.expected_train_artifacts"),
    }
)

# %% [markdown]
# ## 4. Train candidate-error scores and fixed Viterbi continuity selection

# %%
summary = run_hmm_exp226_candidate_selector_on_exp183(
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
        "best_metric": summary["best_metric"],
        "best_viterbi_variant": summary["best_viterbi_variant"],
        "candidate_readout": summary["candidate_readout"],
        "decision": summary["decision"],
    }
)

# %% [markdown]
# ## 5. Metrics, diagnostics, and generated artifacts

# %%
metrics = pd.read_csv(paths.artifacts_dir / summary["artifacts"]["metrics"])
distribution = pd.read_csv(paths.artifacts_dir / summary["artifacts"]["selection_distribution"])
by_well = pd.read_csv(paths.artifacts_dir / summary["artifacts"]["by_well"])
bucket_metrics = pd.read_csv(paths.artifacts_dir / summary["artifacts"]["bucket_metrics"])
subgroup_metrics = pd.read_csv(paths.artifacts_dir / summary["artifacts"]["subgroup_metrics"])
importance = pd.read_csv(paths.artifacts_dir / summary["artifacts"]["feature_importance_mean"])
score_summary = pd.read_csv(paths.artifacts_dir / summary["artifacts"]["score_summary"])
candidate_readout = pd.read_csv(paths.artifacts_dir / summary["artifacts"]["candidate_readout"])
residual_correlation = pd.read_csv(
    paths.artifacts_dir / summary["artifacts"]["residual_correlation"]
)

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

print("distance / stress bucket metrics")
display(bucket_metrics.head(160))

print("subgroup metrics")
display(subgroup_metrics.head(160))

print("feature importance mean")
display(importance.head(120))

print("score summary")
display(score_summary.head(80))

print("candidate readout and residual correlation")
display(candidate_readout)
display(residual_correlation.head(100))

print("generated artifacts")
for key, value in summary["artifacts"].items():
    print(f"{key}: {value}")
