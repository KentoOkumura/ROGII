# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---
# %% [markdown]
# # exp186_typewell_late_range_pfbeam_generation_soft_prior train
#
# CPU-only full replay train feature cache rebuild. Raw competition well/typewell files are the generation input; an existing full replay cache is not read.

# %% [markdown]
# ## Contents
# 1. Setup and configuration
# 2. Raw train input check
# 3. Selected soft-prior contract
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


def selected_soft_prior(config: dict) -> dict:
    selected = cfg_get(
        config,
        "feature_cache.selected_soft_prior_variant",
        cfg_get(config, "model.selected_soft_prior_variant"),
    )
    variants = cfg_get(config, "model.soft_prior_variants", [])
    for variant in variants:
        if variant.get("name") == selected:
            return dict(variant)
    raise ValueError(f"selected soft-prior variant not found: {selected}")


# %%
paths = ExperimentPaths()
paths.require_kaggle_runtime()
paths.ensure_output_dirs()
config = load_config()

feature_cache = cfg_get(config, "feature_cache", {})
soft_prior = selected_soft_prior(config)

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
        "soft_prior": soft_prior,
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
# ## 3. Selected soft-prior contract

# %%
variant_frame = pd.DataFrame(cfg_get(config, "model.soft_prior_variants", []))
display(variant_frame)

required_prior_keys = {
    "name",
    "weak_pct",
    "strong_pct",
    "weak_penalty",
    "strong_penalty",
    "known_last_pct_threshold",
    "known_last_multiplier",
}
missing_prior_keys = sorted(required_prior_keys - set(soft_prior))
if missing_prior_keys:
    raise ValueError(f"selected soft-prior variant is missing keys: {missing_prior_keys}")
if soft_prior["name"] == "no_prior":
    raise ValueError("exp186 must generate a non-baseline soft-prior replay cache")

print("Selected soft-prior variant:", soft_prior["name"])

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
    soft_prior=soft_prior,
)
print(json.dumps(summary, indent=2))

expected = int(feature_cache.get("expected_feature_count", 196))
actual = int(summary["feature_count"])
if actual != expected:
    raise ValueError(f"feature_count mismatch: {actual} != {expected}")

# %% [markdown]
# ## 5. Generated artifacts

# %%
schema_path = paths.artifacts_dir / "exp186_typewell_late_range_pfbeam_generation_soft_prior_full_replay_cache_feature_schema.csv"
summary_path = paths.artifacts_dir / "exp186_typewell_late_range_pfbeam_generation_soft_prior_full_replay_cache_summary.json"
feature_path = (
    paths.artifacts_dir
    / "exp186_typewell_late_range_pfbeam_generation_soft_prior_full_replay_cache_pixiux_likpf_late_soft_prior_public_replay_train_features.csv.gz"
)

schema = pd.read_csv(schema_path)
display(schema.head())
print("Feature cache:", feature_path, "exists=", feature_path.exists())
print("Feature schema:", schema_path, "exists=", schema_path.exists())
print("Summary:", summary_path, "exists=", summary_path.exists())
