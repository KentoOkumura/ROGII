# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---
# %% [markdown]
# # exp196_typewell_late_range_hard_window_pct40_full_cache_replacement train
#
# CPU-only full replay train feature cache rebuild. Raw competition well/typewell files are the generation input; an existing full replay cache is not read.

# %% [markdown]
# ## Contents
# 1. Setup and configuration
# 2. Raw train input check
# 3. Hard-window contract
# 4. Train feature cache generation
# 5. Generated artifacts

# %% [markdown]
# ## 1. Setup and configuration

# %%
from __future__ import annotations

import json

import pandas as pd
from IPython.display import display

from feature_cache import run_train_feature_cache
from settings import ExperimentPaths, get_nested, load_config


def cfg_get(config: dict, dotted_key: str, default=None):
    value = get_nested(config, dotted_key)
    return default if value is None else value


def hard_window_config(config: dict) -> dict:
    value = cfg_get(config, "model.hard_window", {})
    if not isinstance(value, dict):
        raise TypeError("model.hard_window must be a mapping")
    return dict(value)


# %%
paths = ExperimentPaths()
paths.require_kaggle_runtime()
paths.ensure_output_dirs()
config = load_config()

feature_cache = cfg_get(config, "feature_cache", {})
hard_window = hard_window_config(config)

print("Experiment:", cfg_get(config, "experiment.name"))
print("Route:", cfg_get(config, "experiment.route"))
print("Feature cache variant:", feature_cache.get("variant"))
print("Expected feature count:", feature_cache.get("expected_feature_count"))
print("Kaggle GPU enabled:", cfg_get(config, "runtime.kaggle.enable_gpu"))
display(
    {
        "parent": cfg_get(config, "lineage.parent"),
        "implementation_source": cfg_get(config, "lineage.implementation_source"),
        "public_replay_source": cfg_get(config, "lineage.public_replay_source"),
        "output_prefix": feature_cache.get("output_prefix"),
        "n_jobs": feature_cache.get("n_jobs"),
        "pf_seeds": feature_cache.get("pf_seeds"),
        "pf_particles": feature_cache.get("pf_particles"),
        "max_wells": feature_cache.get("max_wells"),
        "hard_window": hard_window,
    }
)

# %% [markdown]
# ## 2. Raw train input check

# %%
train_files = sorted(paths.train_data_dir.glob("*__horizontal_well.csv"))
typewell_files = sorted(paths.train_data_dir.glob("*__typewell.csv"))

if not train_files:
    raise FileNotFoundError(f"No horizontal well files found: {paths.train_data_dir}")
if not typewell_files:
    raise FileNotFoundError(f"No typewell files found: {paths.train_data_dir}")

print("Raw data dir:", paths.raw_data_dir)
print("Train dir:", paths.train_data_dir)
print("Horizontal wells:", len(train_files))
print("Typewells:", len(typewell_files))
print("First train files:", [path.name for path in train_files[:5]])

# %% [markdown]
# ## 3. Hard-window contract

# %%
display(pd.DataFrame([hard_window]))

required_window_keys = {
    "name",
    "enabled",
    "min_typewell_pct",
    "max_typewell_pct",
    "min_rows_after_filter",
}
missing_window_keys = sorted(required_window_keys - set(hard_window))
if missing_window_keys:
    raise ValueError(f"hard-window config is missing keys: {missing_window_keys}")
if hard_window["name"] != "pct40":
    raise ValueError(f"exp196 runs only pct40, got {hard_window['name']}")
if not hard_window["enabled"]:
    raise ValueError("exp196 hard-window filtering must be enabled")
if float(hard_window["min_typewell_pct"]) != 0.40:
    raise ValueError("exp196 must use min_typewell_pct == 0.40")

print("Selected hard-window:", hard_window)

# %% [markdown]
# ## 4. Train feature cache generation

# %%
summary = run_train_feature_cache(
    data_dir=paths.raw_data_dir,
    output_dir=paths.artifacts_dir,
    n_jobs=int(feature_cache.get("n_jobs", 8)),
    pf_seeds=int(feature_cache.get("pf_seeds", 128)),
    pf_particles=int(feature_cache.get("pf_particles", 500)),
    fast=bool(feature_cache.get("fast", False)),
    max_wells=feature_cache.get("max_wells"),
    hard_window=hard_window,
)
print(json.dumps(summary, indent=2))

expected = int(feature_cache.get("expected_feature_count", 196))
actual = int(summary["feature_count"])
if actual != expected:
    raise ValueError(f"feature_count mismatch: {actual} != {expected}")

# %% [markdown]
# ## 5. Generated artifacts

# %%
output_prefix = str(feature_cache["output_prefix"])
variant = str(feature_cache["variant"])
schema_path = paths.artifacts_dir / f"{output_prefix}_feature_schema.csv"
summary_path = paths.artifacts_dir / f"{output_prefix}_summary.json"
feature_path = paths.artifacts_dir / f"{output_prefix}_{variant}_train_features.csv.gz"

schema = pd.read_csv(schema_path)
display(schema.head())
print("Feature cache:", feature_path, "exists=", feature_path.exists())
print("Feature schema:", schema_path, "exists=", schema_path.exists())
print("Summary:", summary_path, "exists=", summary_path.exists())
