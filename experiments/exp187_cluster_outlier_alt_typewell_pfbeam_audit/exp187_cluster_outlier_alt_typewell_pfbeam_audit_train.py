# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---
# %% [markdown]
# # exp187_cluster_outlier_alt_typewell_pfbeam_audit train

# %% [markdown]
# ## Contents
# 1. Imports
# 2. Runtime and configuration helpers
# 3. Input and cluster-strategy contract checks
# 4. Alt typewell PF / Beam generation audit
# 5. Metrics, diagnostics, and generated artifacts

# %% [markdown]
# ## 1. Imports

# %%
from __future__ import annotations

from cluster_outlier_alt_typewell_pfbeam_audit import (
    build_cluster_features,
    run_alt_typewell_pfbeam_audit,
    select_target_wells,
    summarize_strategy_sources,
)
from IPython.display import display
from settings import ExperimentPaths, get_nested, load_config

# %% [markdown]
# ## 2. Runtime and configuration helpers

# %%
paths = ExperimentPaths()
config = load_config()
paths.require_kaggle_runtime()
paths.ensure_output_dirs()

experiment_name = get_nested(config, "experiment.name")
route = get_nested(config, "experiment.route")
audit_config = get_nested(config, "audit") or {}
cluster_config = get_nested(config, "cluster") or {}
validation_surface_config = get_nested(config, "model.validation_surface") or {}
composite_typewell_config = get_nested(config, "model.composite_typewell") or {}
runtime_config = get_nested(config, "model.runtime") or {}
beam_config = get_nested(config, "model.beam") or {}
kaggle_runtime = get_nested(config, "runtime.kaggle") or {}

print(f"experiment={experiment_name}")
print(f"route={route}")
print(f"train_dir={paths.train_data_dir}")
print(f"artifacts_dir={paths.artifacts_dir}")
display(
    {
        "parent": get_nested(config, "lineage.parent"),
        "references": get_nested(config, "lineage.references"),
        "mode": audit_config.get("mode"),
        "gpu_enabled": kaggle_runtime.get("enable_gpu"),
        "validation_surface": validation_surface_config,
        "composite_typewell": composite_typewell_config,
        "pf_runtime": runtime_config,
        "beam": beam_config,
        "cluster": cluster_config,
        "typewell_strategies": get_nested(config, "model.typewell_strategies"),
    }
)

# %% [markdown]
# ## 3. Input and cluster-strategy contract checks

# %%
train_dir = paths.train_data_dir
horizontal_files = sorted(train_dir.glob("*__horizontal_well.csv"))
typewell_files = sorted(train_dir.glob("*__typewell.csv"))

if not horizontal_files:
    raise FileNotFoundError(f"No horizontal well files found: {train_dir}")
if not typewell_files:
    raise FileNotFoundError(f"No typewell files found: {train_dir}")

display(
    {
        "horizontal_wells": len(horizontal_files),
        "typewells": len(typewell_files),
        "first_horizontal": str(horizontal_files[0]),
        "first_typewell": str(typewell_files[0]),
        "score_rows": get_nested(config, "validation.score_rows"),
        "primary_baseline": audit_config.get("primary_baseline"),
    }
)

cluster_features, cluster_meta = build_cluster_features(config)
target_features = select_target_wells(cluster_features, train_dir, config)
strategy_sources = summarize_strategy_sources(target_features, train_dir, config)

print("cluster feature metadata")
display(cluster_meta)

print("target well selection")
display(
    {
        "cluster_feature_wells": int(cluster_features["cluster_feature_valid"].sum()),
        "selected_target_wells": len(target_features),
        "strategy_rows": len(strategy_sources),
        "score_rows": get_nested(config, "validation.score_rows"),
        "exp072_train_feature_cache": get_nested(config, "data.exp072_train_feature_cache_local"),
        "target_gate_counts": target_features["target_gate"].value_counts(dropna=False).to_dict()
        if len(target_features)
        else {},
    }
)

target_preview_cols = [
    "well",
    "cluster_id",
    "own_cluster_dist_z",
    "nearest_other_cluster_id",
    "nearest_other_closer",
    "nearest_other_cluster_rep_well",
    "nearby_weighted_majority_cluster_k8",
    "nearby_weighted_majority_share_k8",
    "nearby_weighted_majority_diff_k8",
    "nearby_weighted_majority_rep_well_k8",
    "target_outlier_score",
]
display(target_features[[col for col in target_preview_cols if col in target_features]].head(80))

print("strategy source preview")
display(strategy_sources.head(120))

if target_features.empty:
    raise RuntimeError("No target wells selected for alt typewell PF/Beam audit.")
if "own_typewell" not in set(strategy_sources["strategy"]):
    raise RuntimeError("own_typewell strategy is missing from selected targets.")
if not (set(strategy_sources["strategy"]) - {"own_typewell"}):
    raise RuntimeError("No alternative cluster-composite typewell strategy is available.")

# %% [markdown]
# ## 4. Alt typewell PF / Beam generation audit

# %%
result = run_alt_typewell_pfbeam_audit(config=config, paths=paths)

summary = result["summary"]
candidate_metrics = result["candidate_metrics"]
strategy_delta_metrics = result["strategy_delta_metrics"]
bucket_metrics = result["bucket_metrics"]
by_well = result["by_well"]
group_metrics = result["group_metrics"]
pf_diagnostics = result["pf_diagnostics"]
well_status = result["well_status"]

print("summary")
display(summary)

# %% [markdown]
# ## 5. Metrics, diagnostics, and generated artifacts

# %%
print("candidate metrics")
display(candidate_metrics.head(80))

print("own-vs-alt delta metrics")
display(strategy_delta_metrics.head(80))

print("PF diagnostics")
display(
    pf_diagnostics.groupby("strategy", observed=True)
    .agg(
        wells=("well", "nunique"),
        rows=("rows", "sum"),
        ess_mean=("ess_mean", "mean"),
        resampling_rate=("resampling_rate", "mean"),
        log_likelihood_mean=("log_likelihood_mean", "mean"),
        seed_weight_max=("seed_weight_max", "mean"),
    )
    .reset_index()
)

print("distance bucket metrics")
display(bucket_metrics.sort_values(["candidate", "distance_bucket"]).head(120))

print("cluster / distance group metrics")
display(group_metrics.sort_values(["group", "rmse"]).head(120))

print("worst wells")
display(
    by_well.sort_values(
        ["candidate", "delta_rmse_vs_primary_baseline"],
        ascending=[True, False],
        na_position="last",
    ).head(120)
)

print("well status")
display(
    well_status["status"].value_counts(dropna=False).rename_axis("status").reset_index(name="wells")
)

print("generated artifacts")
for key, value in summary["artifacts"].items():
    print(f"{key}: {value}")
