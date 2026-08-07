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
# # exp248 candidate perturbation augmentation for likelihood ranker — train
#
# Keep the fixed exp237 eleven-candidate OOF surface and add deterministic,
# target-independent candidate-set perturbations only to outer-train candidate-long rows.
# Clean original candidates are the sole outer-validation and Viterbi states.

# %% [markdown]
# ## Contents
#
# 1. Imports and runtime helpers
# 2. Experiment, cost, and leakage contract
# 3. Input and fixed candidate-source checks
# 4. Deterministic augmentation contract
# 5. Outer well GroupKFold likelihood/error training
# 6. Clean OOF metrics and safety guards
# 7. Feature importance, artifacts, and SHA evidence

# %% [markdown]
# ## 1. Imports and runtime helpers

# %%
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from candidate_perturbation_augmentation_for_likelihood_ranker import (
    OUTPUT_PREFIX,
    run_candidate_perturbation_augmentation,
    synthetic_augmentation_contract_test,
)
from IPython.display import display
from settings import EXPERIMENT_NAME, ExperimentPaths, get_nested, load_config

paths = ExperimentPaths()
config = load_config()
paths.require_kaggle_runtime()
paths.ensure_output_dirs()


def artifact_path(summary: dict, key: str) -> Path:
    return paths.artifacts_dir / summary["artifacts"][key]


# %% [markdown]
# ## 2. Experiment, cost, and leakage contract

# %%
variants = [str(value) for value in get_nested(config, "model.active_variants") or []]
candidate_names = [str(item["name"]) for item in get_nested(config, "ranker.candidates") or []]
transforms = [str(value) for value in get_nested(config, "augmentation.enabled_transforms") or []]
cost_contract = {
    "experiment": EXPERIMENT_NAME,
    "route": get_nested(config, "experiment.route"),
    "parent": get_nested(config, "lineage.parent"),
    "variants": variants,
    "objectives": get_nested(config, "model.estimator"),
    "folds": get_nested(config, "model.planned_folds"),
    "lightgbm_configs": get_nested(config, "model.planned_lightgbm_configs"),
    "boosters": get_nested(config, "model.planned_boosters"),
    "control_retraining": get_nested(config, "model.control_retraining"),
    "parent_retraining": get_nested(config, "model.parent_retraining"),
    "runtime": "kaggle_cpu",
    "gpu": get_nested(config, "runtime.kaggle.enable_gpu"),
    "internet": get_nested(config, "runtime.kaggle.enable_internet"),
    "inference": get_nested(config, "inference.mode"),
}
display(cost_contract)

assert variants == ["original_only", "perturbation_augmented"]
assert int(cost_contract["folds"]) == 5
assert int(cost_contract["lightgbm_configs"]) == 4
assert int(cost_contract["boosters"]) == 20
assert cost_contract["parent_retraining"] is False
assert cost_contract["gpu"] is False
assert cost_contract["internet"] is False
assert get_nested(config, "augmentation.keep_clean_original_view") is True
assert get_nested(config, "augmentation.expose_augmentation_metadata_to_model") is False

print("Leakage policy")
for rule in get_nested(config, "validation.leakage_policy") or []:
    print("-", rule)

# %% [markdown]
# ## 3. Input and fixed candidate-source checks
#
# exp099/072/065/109/114/115/209/223/226 are fixed upstream artifacts. The notebook
# resolves and validates their ID/well/target contracts inside the engine; it does not
# regenerate any PF, HMM, Beam, dense, geometry, exp218, or parent selector model.

# %%
input_contract = {
    "candidate_count": len(candidate_names),
    "candidates": candidate_names,
    "exp099_cache": get_nested(config, "data.exp099_train_feature_cache_local"),
    "exp072_dense_cache": get_nested(config, "data.exp072_train_feature_cache_local"),
    "exp065_clusters": get_nested(config, "data.exp065_cluster_assignments_local"),
    "exp109_typewell_prior": get_nested(config, "data.exp109_oof_predictions_local"),
    "exp114_spatial_prior": get_nested(config, "data.exp114_oof_predictions_local"),
    "exp115_hidden_like": get_nested(config, "data.exp115_fold_assignments_local"),
    "exp209_hmm": get_nested(config, "data.exp209_hmm_train_features_local"),
    "exp223_selfgr_hmm": get_nested(config, "data.exp223_train_features_local"),
    "exp226_geometry": get_nested(config, "data.exp226_train_oof_local"),
    "raw_train_dir": str(paths.train_data_dir),
    "expected_artifacts": get_nested(config, "audit.expected_train_artifacts"),
}
display(input_contract)
assert len(candidate_names) == 11
assert paths.train_data_dir.exists()

# %% [markdown]
# ## 4. Deterministic augmentation contract
#
# Each sampled outer-train row receives at most one single-transform candidate-set view.
# The clean view is always retained. SHA256-derived seeds control sampling and transform
# allocation; target/error/oracle values are not inputs to this stage.

# %%
augmentation_contract = {
    "enabled_transforms": transforms,
    "shift_grid_ft": get_nested(config, "augmentation.shift_grid_ft"),
    "drift": get_nested(config, "augmentation.drift"),
    "spread_scale_grid": get_nested(config, "augmentation.spread_scale_grid"),
    "family_groups": get_nested(config, "augmentation.family_groups"),
    "max_train_base_rows_per_fold": get_nested(config, "augmentation.max_train_base_rows_per_fold"),
    "max_valid_base_rows_for_early_stopping": get_nested(
        config, "augmentation.max_valid_base_rows_for_early_stopping"
    ),
    "seed_policy": get_nested(config, "reproducibility.seed_policy"),
}
display(augmentation_contract)
synthetic_contract = synthetic_augmentation_contract_test()
display(synthetic_contract)
assert set(synthetic_contract["transforms"]) == set(transforms)
assert synthetic_contract["deterministic"] is True

# %% [markdown]
# ## 5. Outer well GroupKFold likelihood/error training
#
# For both variants, every outer-valid well is scored only with clean original candidates.
# The augmented variant concatenates one deterministic perturbed view to the same clean
# sampled train rows. It trains a within-10ft classifier and an L1 expected-error regressor.

# %%
summary = run_candidate_perturbation_augmentation(
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
        "candidate_count": summary["candidate_count"],
        "base_feature_count": summary["base_feature_count"],
        "long_feature_count": summary["long_feature_count"],
        "model_count": summary["model_count"],
        "augmentation_inventory_rows": summary["augmentation_inventory_rows"],
        "decision": summary["decision"],
    }
)

# %% [markdown]
# ## 6. Clean OOF metrics and safety guards

# %%
metrics = pd.read_csv(artifact_path(summary, "metrics"))
candidate_metrics = pd.read_csv(artifact_path(summary, "candidate_metrics"))
calibration = pd.read_csv(artifact_path(summary, "calibration"))
topk = pd.read_csv(artifact_path(summary, "topk_coverage"))
margin = pd.read_csv(artifact_path(summary, "margin_calibration"))
by_well = pd.read_csv(artifact_path(summary, "by_well"))
buckets = pd.read_csv(artifact_path(summary, "bucket_metrics"))
subgroups = pd.read_csv(artifact_path(summary, "subgroup_metrics"))

print("Selected-path metrics")
display(metrics)
print("Candidate likelihood/error calibration metrics")
display(candidate_metrics)
print("Top-K clean candidate coverage")
display(topk)
print("Probability and expected-error calibration")
display(calibration)
print("Top1/top2 expected-error margin calibration")
display(margin)
print("Distance and stress buckets")
display(buckets)
print("Hidden-like and cluster-outlier subgroups")
display(subgroups)
print("Worst wells")
display(by_well.sort_values("rmse_tvt", ascending=False).head(80))

print("Predeclared adoption guard")
print(json.dumps(summary["decision"], indent=2))

# %% [markdown]
# ## 7. Feature importance, artifacts, and SHA evidence

# %%
importance = pd.read_csv(artifact_path(summary, "feature_importance_mean"))
selected_importance = importance[
    importance["variant"].eq("perturbation_augmented")
    & importance["objective"].eq("expected_error_regressor")
].head(30)
display(selected_importance)

if len(selected_importance):
    ax = selected_importance.sort_values("importance").plot.barh(
        x="feature",
        y="importance",
        figsize=(9, 9),
        legend=False,
        title="exp248 augmented expected-error feature importance",
    )
    ax.set_xlabel("mean LightGBM split importance")
    plt.tight_layout()
    plot_path = paths.artifacts_dir / f"{OUTPUT_PREFIX}_feature_importance_top.png"
    plt.savefig(plot_path, dpi=140)
    plt.show()

print("Generated artifacts")
for key, filename in summary["artifacts"].items():
    print(f"{key}: {filename}")

print("SHA256 evidence; .csv.gz values are decompressed-content SHA")
display(
    pd.DataFrame([{"artifact": key, "sha256": value} for key, value in summary["sha256"].items()])
)
