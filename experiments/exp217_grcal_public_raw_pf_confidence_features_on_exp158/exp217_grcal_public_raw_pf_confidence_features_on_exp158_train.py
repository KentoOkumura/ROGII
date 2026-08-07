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
# # exp217_grcal_public_raw_pf_confidence_features_on_exp158 train
#
# Add-only public-like raw PF confidence features for the exp157/158 PF/Beam
# candidate selector. Public raw PF outputs are regenerated on the full train
# selector surface and used only as confidence features; no direct replacement,
# blend, PF-weight replacement, inference port, or submission candidate is
# created.

# %% [markdown]
# ## Contents
#
# 1. Imports
# 2. Runtime and configuration helpers
# 3. Input and feature contract checks
# 4. Load pubraw PF cache, train selector scores, and run Viterbi variants
# 5. Metrics, diagnostics, and generated artifacts

# %% [markdown]
# ## 1. Imports

# %%
from __future__ import annotations

from IPython.display import display

from grcal_public_raw_pf_confidence_features_on_exp158 import (
    run_grcal_public_raw_pf_confidence_features_on_exp158,
)
from settings import EXPERIMENT_NAME, ExperimentPaths, get_nested, load_config

# %% [markdown]
# ## 2. Runtime and configuration helpers

# %%
paths = ExperimentPaths()
config = load_config()
paths.require_kaggle_runtime()
paths.ensure_output_dirs()

pubraw_config = get_nested(config, "ranker.public_raw_pf_features") or {}
selector_config = get_nested(config, "selector") or {}
runtime_config = get_nested(config, "runtime.kaggle") or {}
model_config = get_nested(config, "model") or {}
pf_runtime = get_nested(config, "model.runtime") or {}

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
print("Public raw PF parent:", get_nested(config, "lineage.public_raw_pf_parent"))
print("Artifacts:", paths.artifacts_dir)
print("GPU enabled:", runtime_config.get("enable_gpu"))
print("Candidates:", candidates)
print("Public raw PF particles:", pf_runtime.get("particles"))
print("Public raw PF seed count:", pf_runtime.get("seed_count"))
print("Public raw PF scales:", pf_runtime.get("likelihood_scales"))
print("Public raw PF max target wells:", get_nested(config, "model.validation_surface.max_target_wells"))
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
            "exp115": get_nested(config, "data.exp115_fold_assignments_local"),
        },
        "public_raw_pf_feature_policy": {
            "enabled": pubraw_config.get("enabled"),
            "prefix": pubraw_config.get("prefix"),
            "base_feature_columns": pubraw_config.get("base_feature_columns"),
            "particles": pf_runtime.get("particles"),
            "seed_count": pf_runtime.get("seed_count"),
            "likelihood_scales": pf_runtime.get("likelihood_scales"),
            "forbidden_sources": [
                "exp214 scoped row_candidates join",
                "evaluation-tail true TVT",
                "oracle labels",
                "true-error rank",
            ],
        },
        "historical_baselines": selector_config.get("historical_baselines"),
        "expected_train_artifacts": get_nested(config, "audit.expected_train_artifacts"),
    }
)

# %% [markdown]
# ## 4. Load pubraw PF cache, train selector scores, and run Viterbi variants

# %%
summary = run_grcal_public_raw_pf_confidence_features_on_exp158(
    output_dir=paths.artifacts_dir,
    cache_path=get_nested(config, "data.exp099_train_feature_cache_local"),
    schema_path=get_nested(config, "data.exp099_train_feature_schema_local"),
    max_rows=get_nested(config, "ranker.max_rows"),
)
pubraw_meta = summary["public_raw_pf_features"]

display(
    {
        "status": summary["status"],
        "runtime_seconds": summary["runtime_seconds"],
        "rows": summary["rows"],
        "wells": summary["wells"],
        "feature_count": summary["feature_count"],
        "pubraw_source_mode": pubraw_meta.get("source_mode", "generated"),
        "pubraw_feature_count": pubraw_meta["generated_feature_count"],
        "pubraw_rows": pubraw_meta.get("rows_generated", pubraw_meta.get("rows_loaded")),
        "pubraw_target_wells_generated": pubraw_meta.get("target_wells_generated"),
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
pubraw_summary = pd.read_csv(paths.artifacts_dir / summary["artifacts"]["pubraw_feature_summary"])

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

print("distance / public raw PF / stress bucket metrics")
display(bucket_metrics.head(200))

print("subgroup metrics")
display(subgroup_metrics.head(200))

print("feature importance mean")
display(importance.head(140))

print("score summary")
display(score_summary.head(80))

print("public raw PF feature summary")
display(pubraw_summary.head(120))

print("generated artifacts")
for key, value in summary["artifacts"].items():
    print(f"{key}: {value}")
