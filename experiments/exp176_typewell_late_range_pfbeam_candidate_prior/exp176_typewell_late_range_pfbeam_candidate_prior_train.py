# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---
# %% [markdown]
# # exp176_typewell_late_range_pfbeam_candidate_prior train

# %% [markdown]
# ## Contents
# 1. Imports
# 2. Runtime and configuration helpers
# 3. Input and feature contract
# 4. Candidate prior setup
# 5. Train-side ranker audit
# 6. Metrics and generated artifacts

# %%
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from IPython.display import display
from settings import ExperimentPaths, get_nested, is_kaggle_runtime, load_config
from typewell_late_range_pfbeam_candidate_prior import (
    build_required_columns,
    candidate_specs_from_config,
    find_artifact,
    read_typewell_late_range_context,
    run_typewell_late_range_pfbeam_candidate_prior,
)

# %% [markdown]
# ## 2. Runtime and configuration helpers

# %%
paths = ExperimentPaths()
config = load_config()
output_dir = Path("/kaggle/working/artifacts") if is_kaggle_runtime() else paths.artifacts_dir
output_dir.mkdir(parents=True, exist_ok=True)

experiment = config["experiment"]
ranker = config["ranker"]
print("experiment:", experiment["name"])
print("route:", experiment["route"])
print("status:", experiment["status"])
print("output_dir:", output_dir)
print("kaggle_runtime:", is_kaggle_runtime())

# %% [markdown]
# ## 3. Input and feature contract

# %%
candidates = candidate_specs_from_config(config)
required_columns = build_required_columns(config, candidates)
cache_path = get_nested(config, "data.exp099_train_feature_cache_local")
schema_path = get_nested(config, "data.exp099_train_feature_schema_local")
dense_cache_path = get_nested(config, "data.exp072_train_feature_cache_local")
dense_schema_path = get_nested(config, "data.exp072_feature_schema_local")

cache_file = find_artifact(
    "exp099_pf_multi_observation_likelihood_probe_multiobs_likelihood_probe_train_features.csv.gz",
    cache_path,
)
schema_file = find_artifact(
    "exp099_pf_multi_observation_likelihood_probe_multiobs_likelihood_probe_feature_schema.csv",
    schema_path,
)
dense_cache_file = find_artifact(
    "exp063_full_replay_feature_cache_pixiux_likpf_public_replay_train_features.csv.gz",
    dense_cache_path,
)
dense_schema_file = find_artifact(
    "exp063_full_replay_feature_cache_feature_schema.csv",
    dense_schema_path,
)

print("candidate_count:", len(candidates), [spec.name for spec in candidates])
print("required_source_columns:", len(required_columns))
print("main_cache:", cache_file)
print("main_schema:", schema_file)
print("dense_cache:", dense_cache_file)
print("dense_schema:", dense_schema_file)

header = pd.read_csv(cache_file, nrows=0).columns.tolist()
missing = [column for column in required_columns if column not in header]
if missing:
    raise RuntimeError(f"main cache missing required columns: {missing}")
print("main_cache_header_columns:", len(header))

# %% [markdown]
# ## 4. Candidate prior setup

# %%
prior = get_nested(config, "ranker.typewell_late_range_prior") or {}
context, context_meta = read_typewell_late_range_context(
    train_dir=paths.train_data_dir,
    min_typewell_span=float(prior.get("min_typewell_span", 1.0)),
)
print(json.dumps(context_meta, indent=2, sort_keys=True))
display(
    context[["well", "typewell_min", "typewell_max", "typewell_span", "known_last_pct"]]
    .sort_values("known_last_pct", ascending=False)
    .head(10)
)
print("known_last_pct_min:", prior.get("known_last_pct_min"))
print("candidate_pct_lower_bounds:", prior.get("candidate_pct_lower_bounds"))
print("known_last_margins:", prior.get("known_last_margins"))

# %% [markdown]
# ## 5. Train-side ranker audit

# %%
summary = run_typewell_late_range_pfbeam_candidate_prior(
    output_dir=output_dir,
    cache_path=cache_path,
    schema_path=schema_path,
    max_rows=get_nested(config, "ranker.max_rows"),
)

# %% [markdown]
# ## 6. Metrics and generated artifacts

# %%
metrics_path = output_dir / summary["artifacts"]["metrics"]
importance_path = output_dir / summary["artifacts"]["feature_importance_mean"]
schema_out_path = output_dir / summary["artifacts"]["feature_schema"]

metrics = pd.read_csv(metrics_path)
display(metrics.sort_values("rmse_tvt"))

importance = pd.read_csv(importance_path)
display(
    importance.sort_values(["variant", "mean_importance"], ascending=[True, False])
    .groupby("variant", observed=True)
    .head(20)
)

print("feature_count:", summary["feature_count"])
print("summary_decision:", json.dumps(summary["decision"], indent=2, sort_keys=True))
print("metrics:", metrics_path)
print("feature_importance:", importance_path)
print("feature_schema:", schema_out_path)
print("summary_sha:", json.dumps(summary["sha256"], indent=2, sort_keys=True))
