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
# # exp217_grcal_public_raw_pf_confidence_features_on_exp158 pubraw cache
#
# Generate the public-like raw PF confidence feature cache used by exp217.
# This notebook does not train LightGBM models and does not create a submission.

# %% [markdown]
# ## Contents
#
# 1. Imports
# 2. Runtime and configuration helpers
# 3. Input and cache contract checks
# 4. Generate pubraw PF cache
# 5. Generated artifacts

# %% [markdown]
# ## 1. Imports

# %%
from __future__ import annotations

from IPython.display import display

from grcal_public_raw_pf_confidence_features_on_exp158 import (
    run_public_raw_pf_cache_generation,
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
pubraw_cache = pubraw_config.get("cache") if isinstance(pubraw_config.get("cache"), dict) else {}
runtime_config = get_nested(config, "runtime.kaggle") or {}
pf_runtime = get_nested(config, "model.runtime") or {}

print("Experiment:", EXPERIMENT_NAME)
print("Route:", get_nested(config, "experiment.route"))
print("Mode:", get_nested(config, "audit.mode"))
print("Artifacts:", paths.artifacts_dir)
print("GPU enabled:", runtime_config.get("enable_gpu"))
print("Public raw PF particles:", pf_runtime.get("particles"))
print("Public raw PF seed count:", pf_runtime.get("seed_count"))
print("Public raw PF scales:", pf_runtime.get("likelihood_scales"))
print("Public raw PF max target wells:", get_nested(config, "model.validation_surface.max_target_wells"))
print("Pubraw cache file:", pubraw_cache.get("features_file"))

# %% [markdown]
# ## 3. Input and cache contract checks

# %%
display(
    {
        "data_sources": {
            "exp099": get_nested(config, "data.exp099_train_feature_cache_local"),
            "exp072": get_nested(config, "data.exp072_train_feature_cache_local"),
        },
        "cache_policy": {
            "features_file": pubraw_cache.get("features_file"),
            "summary_file": pubraw_cache.get("summary_file"),
            "source_kernel": pubraw_cache.get("source_kernel"),
            "read_cache_in_downstream_train": pubraw_cache.get("enabled"),
        },
        "public_raw_pf_feature_policy": {
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
    }
)

# %% [markdown]
# ## 4. Generate pubraw PF cache

# %%
summary = run_public_raw_pf_cache_generation(
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
        "generated_feature_count": summary["generated_feature_count"],
        "artifacts": summary["artifacts"],
        "sha256": summary["sha256"],
        "pubraw_rows_generated": summary["public_raw_pf_features"]["rows_generated"],
        "pubraw_target_wells_generated": summary["public_raw_pf_features"][
            "target_wells_generated"
        ],
    }
)

# %% [markdown]
# ## 5. Generated artifacts

# %%
import pandas as pd

feature_summary = pd.read_csv(paths.artifacts_dir / summary["artifacts"]["pubraw_cache_feature_summary"])
schema = pd.read_csv(paths.artifacts_dir / summary["artifacts"]["pubraw_cache_schema"])

print("pubraw cache feature summary")
display(feature_summary.head(120))

print("pubraw cache schema")
display(schema.head(120))
