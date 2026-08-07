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
# # exp175_cluster_outlier_typewell_prior_gate train
#
# No-training OOF audit for weak typewell/spatial prior corrections
# on native typewell cluster outlier wells.

# %% [markdown]
# ## Contents
#
# 1. Imports
# 2. Runtime and configuration helpers
# 3. Input and fixed source checks
# 4. Cluster-outlier prior gate audit
# 5. Metrics, diagnostics, and generated artifacts

# %% [markdown]
# ## 1. Imports

# %%
from __future__ import annotations

from IPython.display import display

from cluster_outlier_typewell_prior_gate import run_cluster_outlier_typewell_prior_gate
from settings import EXPERIMENT_NAME, ExperimentPaths, get_nested, load_config

# %% [markdown]
# ## 2. Runtime and configuration helpers

# %%
paths = ExperimentPaths()
config = load_config()
paths.require_kaggle_runtime()
paths.ensure_output_dirs()

prediction_sources = get_nested(config, "prediction_sources") or []
prior_variants = get_nested(config, "gate.prior_variants") or []
cluster_gates = get_nested(config, "cluster.gates") or []
quality_gates = get_nested(config, "gate.prior_quality") or []
alphas = get_nested(config, "gate.correction_alphas") or []
clips = get_nested(config, "gate.correction_clip_ft") or []
runtime_config = get_nested(config, "runtime.kaggle") or {}

policy_grid_count = (
    len(prediction_sources)
    * len(prior_variants)
    * len(cluster_gates)
    * len(quality_gates)
    * len(alphas)
    * len(clips)
)

print("Experiment:", EXPERIMENT_NAME)
print("Route:", get_nested(config, "experiment.route"))
print("Mode:", get_nested(config, "audit.mode"))
print("Parent:", get_nested(config, "lineage.parent"))
print("Fallback parent:", get_nested(config, "lineage.fallback_parent"))
print("Artifacts:", paths.artifacts_dir)
print("GPU enabled:", runtime_config.get("enable_gpu"))
print("Prediction sources:", [source.get("name") for source in prediction_sources])
print("Prior variants:", [prior.get("name") for prior in prior_variants])
print("Cluster gates:", [gate.get("name") for gate in cluster_gates])
print("Quality gates:", [gate.get("name") for gate in quality_gates])
print("Posthoc policies:", policy_grid_count)
print("LightGBM configs: 0 folds: 0 boosters: 0 control retraining: none")

# %% [markdown]
# ## 3. Input and fixed source checks

# %%
display(
    {
        "cluster_assignment_method": get_nested(config, "cluster.assignment_method"),
        "cluster_assignment_threshold": get_nested(config, "cluster.assignment_threshold"),
        "min_cluster_size": get_nested(config, "cluster.min_cluster_size"),
        "outlier_z_thresholds": get_nested(config, "cluster.outlier_z_thresholds"),
        "nearby_k_values": get_nested(config, "cluster.nearby_k_values"),
        "correction_alphas": alphas,
        "correction_clip_ft": clips,
        "expected_train_artifacts": get_nested(config, "audit.expected_train_artifacts"),
    }
)

# %% [markdown]
# ## 4. Cluster-outlier prior gate audit

# %%
summary = run_cluster_outlier_typewell_prior_gate(
    config=config,
    paths=paths,
)

display(
    {
        "decision": summary["decision"],
        "runtime_seconds": summary["runtime_seconds"],
        "rows": summary["rows"],
        "wells": summary["wells"],
        "best_policy": summary["best_policy"],
        "best_by_well_delta": summary["best_by_well_delta"],
    }
)

# %% [markdown]
# ## 5. Metrics, diagnostics, and generated artifacts

# %%
import pandas as pd

gate_metrics = pd.read_csv(summary["artifacts"]["gate_metrics"])
by_well_delta = pd.read_csv(summary["artifacts"]["by_well_delta"])
bucket_metrics = pd.read_csv(summary["artifacts"]["bucket_metrics"])
subgroup_metrics = pd.read_csv(summary["artifacts"]["subgroup_metrics"])
cluster_features = pd.read_csv(summary["artifacts"]["cluster_outlier_well_features"])

print("gate metrics")
display(gate_metrics.head(80))

print("worst well delta")
display(
    by_well_delta.sort_values(
        ["max_regression_rmse", "mean_delta_rmse"],
        ascending=[False, False],
    ).head(80)
)

print("distance bucket metrics")
display(bucket_metrics.sort_values(["policy", "distance_bucket"]).head(120))

print("subgroup metrics")
display(subgroup_metrics.sort_values(["policy", "subgroup"]).head(160))

print("cluster outlier feature summary")
display(
    cluster_features[
        [
            "well",
            "cluster_id",
            "cluster_size",
            "own_cluster_dist",
            "own_cluster_dist_z",
            "nearest_other_cluster_id",
            "nearest_other_cluster_dist",
            "nearest_other_closer",
            "nearby_majority_cluster_k8",
            "nearby_majority_share_k8",
            "nearby_majority_diff_k8",
        ]
    ]
    .sort_values("own_cluster_dist_z", ascending=False)
    .head(80)
)

print("generated artifacts")
for key, value in summary["artifacts"].items():
    print(f"{key}: {value}")
