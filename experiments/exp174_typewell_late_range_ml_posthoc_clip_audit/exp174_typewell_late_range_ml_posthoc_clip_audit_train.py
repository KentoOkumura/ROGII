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
# # exp174_typewell_late_range_ml_posthoc_clip_audit train
#
# No-training OOF audit for conditional typewell late-range posthoc shrink/clip on fixed ML predictions.

# %% [markdown]
# ## Contents
#
# 1. Imports
# 2. Runtime and configuration helpers
# 3. Input and prediction source checks
# 4. Typewell late-range posthoc audit
# 5. Metrics, diagnostics, and generated artifacts

# %% [markdown]
# ## 1. Imports

# %%
from __future__ import annotations

from IPython.display import display

from settings import EXPERIMENT_NAME, ExperimentPaths, get_nested, load_config
from typewell_late_range_ml_posthoc_clip_audit import (
    run_typewell_late_range_ml_posthoc_clip_audit,
)

# %% [markdown]
# ## 2. Runtime and configuration helpers

# %%
paths = ExperimentPaths()
config = load_config()
paths.require_kaggle_runtime()
paths.ensure_output_dirs()

posthoc = get_nested(config, "model.posthoc") or {}
runtime_config = get_nested(config, "runtime.kaggle") or {}
prediction_sources = get_nested(config, "data.prediction_sources") or []

variant_count = (
    len(posthoc.get("known_last_pct_min", []))
    * (
        len(posthoc.get("fixed_lower_bounds", []))
        + len(posthoc.get("known_last_margins", []))
    )
    * len(posthoc.get("alphas", []))
)

print("Experiment:", EXPERIMENT_NAME)
print("Route:", get_nested(config, "experiment.route"))
print("Mode:", get_nested(config, "audit.mode"))
print("Parent:", get_nested(config, "lineage.parent"))
print("Train data:", paths.train_data_dir)
print("Test data:", paths.test_data_dir)
print("Artifacts:", paths.artifacts_dir)
print("GPU enabled:", runtime_config.get("enable_gpu"))
print("Prediction sources:", [source.get("name") for source in prediction_sources])
print("Posthoc variants:", variant_count)
print("LightGBM configs: 0 folds: 0 boosters: 0 control retraining: none")

# %% [markdown]
# ## 3. Input and prediction source checks

# %%
horizontal_files = sorted(paths.train_data_dir.glob("*__horizontal_well.csv"))
typewell_files = sorted(paths.train_data_dir.glob("*__typewell.csv"))
test_horizontal_files = sorted(paths.test_data_dir.glob("*__horizontal_well.csv"))

if not horizontal_files:
    raise FileNotFoundError(f"No train horizontal well files found: {paths.train_data_dir}")
if not typewell_files:
    raise FileNotFoundError(f"No train typewell files found: {paths.train_data_dir}")

display(
    {
        "train_horizontal_wells": len(horizontal_files),
        "train_typewells": len(typewell_files),
        "test_horizontal_wells": len(test_horizontal_files),
        "first_train_horizontal": str(horizontal_files[0]),
        "first_train_typewell": str(typewell_files[0]),
        "known_last_pct_min": posthoc.get("known_last_pct_min"),
        "fixed_lower_bounds": posthoc.get("fixed_lower_bounds"),
        "known_last_margins": posthoc.get("known_last_margins"),
        "alphas": posthoc.get("alphas"),
        "save_oof_top_k": posthoc.get("save_oof_top_k"),
    }
)

# %% [markdown]
# ## 4. Typewell late-range posthoc audit

# %%
result = run_typewell_late_range_ml_posthoc_clip_audit(
    config=config,
    train_dir=paths.train_data_dir,
    test_dir=paths.test_data_dir,
    output_dir=paths.artifacts_dir,
    metrics_path=paths.metrics_path,
)

summary = result["summary"]
candidate_metrics = result["candidate_metrics"]
bucket_metrics = result["bucket_metrics"]
by_well_metrics = result["by_well_metrics"]
group_metrics = result["group_metrics"]
changed_summary = result["changed_summary"]
source_summary = result["source_summary"]
typewell_summary = result["typewell_summary"]

display(
    {
        "status": summary["status"],
        "runtime_sec": summary["runtime_sec"],
        "policy_grid_count": summary["policy_grid_count"],
        "source_count_loaded": summary["source_count_loaded"],
        "best_by_source": summary["best_by_source"],
    }
)

# %% [markdown]
# ## 5. Metrics, diagnostics, and generated artifacts

# %%
print("candidate metrics")
display(candidate_metrics.head(80))

print("baseline and top changed summaries")
display(changed_summary.head(80))

print("group metrics")
display(
    group_metrics.sort_values(["source", "group", "rmse_tvt", "policy"]).head(160)
)

print("distance bucket metrics")
display(
    bucket_metrics[bucket_metrics["bucket_family"].eq("distance_bucket")]
    .sort_values(["source", "bucket", "rmse_tvt", "policy"])
    .head(160)
)

print("known-last pct bucket metrics")
display(
    bucket_metrics[bucket_metrics["bucket_family"].eq("known_last_pct_bucket")]
    .sort_values(["source", "bucket", "rmse_tvt", "policy"])
    .head(160)
)

print("worst well regressions")
display(
    by_well_metrics.sort_values(
        ["source", "policy", "rmse_delta_vs_baseline", "rmse_tvt"],
        ascending=[True, True, False, False],
    ).head(160)
)

print("source summary")
display(source_summary)

print("typewell summary")
display(typewell_summary)

print("generated artifacts")
for key, value in summary["artifacts"].items():
    print(f"{key}: {value}")
