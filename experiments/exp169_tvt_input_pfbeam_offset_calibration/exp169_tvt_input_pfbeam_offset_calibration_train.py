# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---
# %% [markdown]
# # exp169_tvt_input_pfbeam_offset_calibration train

# %% [markdown]
# ## Contents
# 1. Imports
# 2. Runtime and configuration helpers
# 3. Input checks
# 4. Prefix holdout replay and offset calibration audit
# 5. Metrics and generated artifacts

# %% [markdown]
# ## 1. Imports

# %%
from __future__ import annotations

import pandas as pd
from IPython.display import display

from settings import ExperimentPaths, get_nested, load_config
from tvt_input_pfbeam_offset_calibration import run_audit

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
offset_config = get_nested(config, "model.offset_calibration") or {}
replay_config = get_nested(config, "model.replay_runtime") or {}

print(f"experiment={experiment_name}")
print(f"route={route}")
print(f"train_dir={paths.train_data_dir}")
print(f"artifacts_dir={paths.artifacts_dir}")
print(f"mode={audit_config.get('mode')}")
display(
    {
        "parent": get_nested(config, "lineage.parent"),
        "cache_parent": get_nested(config, "lineage.cache_parent"),
        "primary_baseline": audit_config.get("primary_baseline"),
        "prefix_holdout_rows": offset_config.get("prefix_holdout_rows"),
        "min_known_prefix_rows": offset_config.get("min_known_prefix_rows"),
        "min_calibration_rows": offset_config.get("min_calibration_rows"),
        "replay_runtime": replay_config,
    }
)

# %% [markdown]
# ## 3. Input checks

# %%
train_dir = paths.train_data_dir
horizontal_files = sorted(train_dir.glob("*__horizontal_well.csv"))
typewell_files = sorted(train_dir.glob("*__typewell.csv"))

if not horizontal_files:
    raise FileNotFoundError(f"No horizontal well files found: {train_dir}")
if not typewell_files:
    raise FileNotFoundError(f"No typewell files found: {train_dir}")

print(f"horizontal wells={len(horizontal_files)}")
print(f"typewells={len(typewell_files)}")
display(
    {
        "first_horizontal": str(horizontal_files[0]),
        "first_typewell": str(typewell_files[0]),
        "exp072_train_feature_cache_local": get_nested(
            config,
            "data.exp072_train_feature_cache_local",
        ),
        "candidate_count": len(audit_config.get("candidates", [])),
        "variant_grid": offset_config.get("correction_grid"),
    }
)

# %% [markdown]
# ## 4. Prefix holdout replay and offset calibration audit

# %%
summary = run_audit(config=config, paths=paths)

print("summary")
display(summary)

# %% [markdown]
# ## 5. Metrics and generated artifacts

# %%
artifacts = summary["artifacts"]

candidate_metrics = pd.read_csv(artifacts["candidate_metrics"])
bucket_metrics = pd.read_csv(artifacts["bucket_metrics"])
group_metrics = pd.read_csv(artifacts["group_metrics"])
prefix_offsets = pd.read_csv(artifacts["prefix_offsets"])
prefix_status = pd.read_csv(artifacts["prefix_status"])

print("candidate metrics")
display(candidate_metrics.head(40))

print("prefix replay status")
display(prefix_status["status"].value_counts().rename_axis("status").reset_index(name="wells"))

print("prefix offset summary")
display(
    prefix_offsets.groupby(["offset_source", "candidate"], observed=True)
    .agg(
        wells=("well", "nunique"),
        rows_median=("rows", "median"),
        offset_median=("offset_median", "median"),
        offset_iqr_median=("offset_iqr", "median"),
        prefix_rmse_median=("prefix_rmse", "median"),
    )
    .reset_index()
    .sort_values(["offset_source", "candidate"])
)

print("distance bucket metrics")
display(bucket_metrics.sort_values(["candidate", "distance_bucket"]).head(120))

print("group metrics")
display(group_metrics.sort_values(["group", "rmse"]).head(120))

print("generated artifacts")
for key, value in artifacts.items():
    print(f"{key}: {value}")
