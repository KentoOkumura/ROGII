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
# # exp251 raw-test-safe dual-objective candidate ranker — train
#
# Re-audit all 297 exp248 original-only candidate-long features against an independently
# reconstructed raw-test surface. The default stage trains no model. The optional train
# stage re-runs the same audit first and fits only the features that pass its provenance
# and fallback contract.

# %% [markdown]
# ## Contents
#
# 1. Imports and runtime helpers
# 2. Experiment, stage, and compute contract
# 3. Input and fixed candidate-source contract
# 4. Raw-test provenance and distribution audit
# 5. Optional dual-objective outer-fold training
# 6. Metrics, feature importance, and safety guards
# 7. Generated artifacts and SHA evidence

# %% [markdown]
# ## 1. Imports and runtime helpers

# %%
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from raw_test_safe_dual_objective_candidate_ranker import (
    OUTPUT_PREFIX,
    run_raw_test_safe_candidate_ranker,
    synthetic_feature_audit_contract_test,
)
from settings import EXPERIMENT_NAME, ExperimentPaths, get_nested, load_config

paths = ExperimentPaths()
config = load_config()
paths.require_kaggle_runtime()
paths.ensure_output_dirs()


def artifact_path(summary: dict, key: str) -> Path:
    return paths.artifacts_dir / summary["artifacts"][key]


# %% [markdown]
# ## 2. Experiment, stage, and compute contract
#
# `feature_audit_only` is the committed default and trains zero boosters.
# `train_after_feature_audit` is implemented but must be explicitly selected before a
# later Kaggle CPU push; it runs one variant, two objectives, five folds, and ten boosters.

# %%
stage = str(get_nested(config, "execution.stage"))
variants = [str(value) for value in get_nested(config, "model.active_variants") or []]
cost_contract = {
    "experiment": EXPERIMENT_NAME,
    "route": get_nested(config, "experiment.route"),
    "parent": get_nested(config, "lineage.parent"),
    "candidate_parent": get_nested(config, "lineage.candidate_parent"),
    "stage": stage,
    "variants": variants if stage == "train_after_feature_audit" else [],
    "objectives": get_nested(config, "model.estimator")
    if stage == "train_after_feature_audit"
    else [],
    "folds": get_nested(config, "model.planned_folds")
    if stage == "train_after_feature_audit"
    else 0,
    "lightgbm_configs": get_nested(config, "model.planned_lightgbm_configs")
    if stage == "train_after_feature_audit"
    else 0,
    "boosters": get_nested(config, "model.planned_boosters")
    if stage == "train_after_feature_audit"
    else 0,
    "control_retraining": get_nested(config, "model.control_retraining"),
    "parent_retraining": get_nested(config, "model.parent_retraining"),
    "runtime": "kaggle_cpu",
    "gpu": get_nested(config, "runtime.kaggle.enable_gpu"),
    "internet": get_nested(config, "runtime.kaggle.enable_internet"),
    "inference": get_nested(config, "inference.mode"),
}
display(cost_contract)

assert stage in {"feature_audit_only", "train_after_feature_audit"}
assert variants == ["raw_test_regenerated_copcf"]
assert int(get_nested(config, "model.planned_lightgbm_configs")) == 2
assert int(get_nested(config, "model.planned_folds")) == 5
assert int(get_nested(config, "model.planned_boosters")) == 10
assert cost_contract["control_retraining"] is False
assert cost_contract["parent_retraining"] is False
assert cost_contract["gpu"] is False
assert cost_contract["internet"] is False
assert get_nested(config, "augmentation.enabled") is False

print("Leakage and raw-test contract")
for rule in get_nested(config, "validation.leakage_policy") or []:
    print("-", rule)

# %% [markdown]
# ## 3. Input and fixed candidate-source contract
#
# Train-side inputs reproduce exp248's fixed eleven-candidate OOF surface. Raw-test inputs
# independently read exp073 base features and exp226 predictions and regenerate exp209 /
# exp223 HMM paths and multi-observation GR features. No candidate or upstream model is fit.

# %%
candidate_names = [str(item["name"]) for item in get_nested(config, "ranker.candidates") or []]
input_contract = {
    "candidate_count": len(candidate_names),
    "candidates": candidate_names,
    "train_feature_cache": get_nested(config, "data.exp099_train_feature_cache_local"),
    "dense_train_cache": get_nested(config, "data.exp072_train_feature_cache_local"),
    "exp209_train_hmm": get_nested(config, "data.exp209_hmm_train_features_local"),
    "exp223_train_hmm": get_nested(config, "data.exp223_train_features_local"),
    "exp226_train_oof": get_nested(config, "data.exp226_train_oof_local"),
    "rawtest_base_cache": get_nested(config, "data.exp073_test_feature_cache_local"),
    "rawtest_exp226_summary": get_nested(config, "data.exp226_test_summary_local"),
    "raw_train_dir": str(paths.train_data_dir),
    "raw_test_dir": str(paths.test_data_dir),
    "sample_submission": str(paths.sample_submission_path),
}
display(input_contract)
assert len(candidate_names) == 11
assert paths.train_data_dir.exists()
assert paths.test_data_dir.exists()
assert paths.sample_submission_path.exists()

# %% [markdown]
# ## 4. Raw-test provenance and distribution audit
#
# The audit records generation availability, train/raw-test missing rates, quantiles,
# standardized mean difference, PSI, provenance, and the final inclusion reason for every
# exp248 long feature. `copcf_*` is cross-fitted on train and independently regenerated
# from full-train references on raw test. Train-only exp226 auxiliaries remain explicit
# rejects; unavailable columns are never rescued with median or zero fallback.

# %%
feature_contract = {
    "expected_parent_features": get_nested(
        config, "feature_audit.expected_parent_long_feature_count"
    ),
    "expected_selected_features": get_nested(
        config, "feature_audit.expected_selected_long_feature_count"
    ),
    "expected_regenerated_prefix_features": get_nested(
        config, "feature_audit.expected_regenerated_prefix_feature_count"
    ),
    "minimum_selected_features": get_nested(
        config, "feature_audit.min_selected_long_feature_count"
    ),
    "max_missing_rate": get_nested(config, "feature_audit.max_missing_rate"),
    "max_missing_rate_delta": get_nested(config, "feature_audit.max_missing_rate_delta"),
    "disallowed_prefixes": get_nested(config, "feature_audit.disallowed_prefixes"),
    "disallowed_exact_columns": get_nested(config, "feature_audit.disallowed_exact_columns"),
    "allowed_provenance": get_nested(config, "feature_audit.allowed_provenance"),
}
display(feature_contract)
display(synthetic_feature_audit_contract_test())

summary = run_raw_test_safe_candidate_ranker(
    output_dir=paths.artifacts_dir,
    cache_path=get_nested(config, "data.exp099_train_feature_cache_local"),
    schema_path=get_nested(config, "data.exp099_train_feature_schema_local"),
)
display(
    {
        "status": summary["status"],
        "execution_stage": summary["execution_stage"],
        "feature_decision": summary.get("feature_decision")
        or summary.get("feature_audit", {}).get("feature_decision"),
        "rawtest_copcf": (summary.get("rawtest_source_meta") or {}).get("copcf"),
        "model_count": summary.get("model_count", 0),
    }
)

audit = pd.read_csv(paths.artifacts_dir / f"{OUTPUT_PREFIX}_feature_audit.csv")
display(audit.groupby(["provenance", "selected"], dropna=False).size().rename("features"))
display(audit.loc[~audit["selected"]].head(100))
display(audit.sort_values(["distribution_warning", "psi"], ascending=[False, False]).head(100))

# %% [markdown]
# ## 5. Optional dual-objective outer-fold training
#
# This section has outputs only when `execution.stage=train_after_feature_audit`. The same
# run must first pass the feature audit. Each outer fold fits a within-10 classifier and
# expected-absolute-error regressor; clean original candidates are the only validation and
# Viterbi states.

# %%
if stage == "feature_audit_only":
    print("Feature-audit-only stage completed: 0 variants / 0 configs / 0 folds / 0 boosters")
else:
    assert summary["model_count"] == 10
    print("Same-run feature audit passed; 10 CPU boosters completed")

# %% [markdown]
# ## 6. Metrics, feature importance, and safety guards

# %%
if stage == "train_after_feature_audit":
    metrics = pd.read_csv(artifact_path(summary, "metrics"))
    candidate_metrics = pd.read_csv(artifact_path(summary, "candidate_metrics"))
    buckets = pd.read_csv(artifact_path(summary, "bucket_metrics"))
    subgroups = pd.read_csv(artifact_path(summary, "subgroup_metrics"))
    by_well = pd.read_csv(artifact_path(summary, "by_well"))
    importance = pd.read_csv(artifact_path(summary, "feature_importance_mean"))
    display(metrics)
    display(candidate_metrics)
    display(buckets.query("bucket == '1000_plus'"))
    display(subgroups[subgroups["subgroup"].astype(str).str.startswith("exp115_")])
    display(by_well.sort_values("rmse_tvt", ascending=False).head(80))
    print(json.dumps(summary["decision"], indent=2))

    selected_importance = importance[importance["objective"].eq("expected_error_regressor")].head(
        30
    )
    display(selected_importance)
    if len(selected_importance):
        ax = selected_importance.sort_values("importance").plot.barh(
            x="feature",
            y="importance",
            figsize=(9, 9),
            legend=False,
            title="exp251 raw-test-safe expected-error feature importance",
        )
        ax.set_xlabel("mean LightGBM split importance")
        plt.tight_layout()
        plt.savefig(paths.artifacts_dir / f"{OUTPUT_PREFIX}_feature_importance_top.png", dpi=140)
        plt.show()

# %% [markdown]
# ## 7. Generated artifacts and SHA evidence

# %%
print("Generated artifacts")
for key, filename in summary["artifacts"].items():
    print(f"{key}: {filename}")

print("SHA256 evidence; gzip feature samples and OOF use decompressed-content SHA")
display(
    pd.DataFrame([{"artifact": key, "sha256": value} for key, value in summary["sha256"].items()])
)
