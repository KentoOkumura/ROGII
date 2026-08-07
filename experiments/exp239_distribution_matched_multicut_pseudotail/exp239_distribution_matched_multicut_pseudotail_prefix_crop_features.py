# %% [markdown]
# # exp239 pseudo-tail exp218 feature cache (CPU)
#
# Generate the 380-column exp218 pseudo-tail training surface in bounded
# request batches. This stage trains no model and uses no GPU.

# %% [markdown]
# ## Contents
# 1. Imports and runtime helpers
# 2. Configuration and input contract
# 3. Deterministic cutoff and fold manifests
# 4. Prefix materialization
# 5. Chunked exp218 feature-cache generation
# 6. Cache manifest and reproducibility summary

# %%
from __future__ import annotations

import importlib
import os
from pathlib import Path

os.environ["EXP239_IMPORT_ONLY"] = "1"

audit = importlib.import_module("exp239_distribution_matched_multicut_pseudotail_train")
augmentation = importlib.import_module("exp239_exp218_pseudotail_augmentation")


# %% [markdown]
# ## 2. Configuration and input contract
#
# The CPU stage keeps the approved evaluation surface unchanged: 800 replay
# requests, 799,961 pseudo rows, and the same 380-feature exp218 schema. It
# writes Parquet shards only; LightGBM config/fold/booster counts are all zero.

# %%
audit.require_allowed_runtime()
CONFIG_PATH = audit.find_named_file(
    "config.yaml",
    [
        Path("experiments") / audit.EXPERIMENT_NAME / "config.yaml",
        Path.cwd() / "config.yaml",
    ],
)
CONFIG = audit.read_yaml(CONFIG_PATH)
TRAIN_DIR = audit.find_train_dir(CONFIG)
TRAIN_FILES = sorted(
    TRAIN_DIR.glob(str(audit.nested(CONFIG, "data.horizontal_glob", "*__horizontal_well.csv")))
)
if not TRAIN_FILES:
    raise FileNotFoundError(f"no horizontal wells found under {TRAIN_DIR}")
print(f"config={CONFIG_PATH}")
print(f"route={audit.nested(CONFIG, 'experiment.route')}")
print("stage=cpu_feature_cache")
print(f"train_dir={TRAIN_DIR}")
print(f"horizontal_wells={len(TRAIN_FILES)}")
print(
    "approved_counts="
    f"requests={audit.nested(CONFIG, 'model.exp218_augmentation.pseudo_request_count')} "
    f"rows={audit.nested(CONFIG, 'model.exp218_augmentation.expected_pseudo_rows')} "
    f"features={audit.nested(CONFIG, 'model.exp218_augmentation.expected_feature_count')} "
    "configs=0 folds=0 boosters=0 control_retrained=false"
)


# %% [markdown]
# ## 3. Deterministic cutoff and fold manifests

# %%
ROLES, ROLES_META = audit.load_hidden_like_roles(CONFIG)
METADATA, FRAMES = audit.build_well_metadata(TRAIN_FILES, ROLES, CONFIG)
CANDIDATES = audit.build_cutoff_candidates(METADATA, FRAMES, CONFIG)
FOLD_MANIFEST = audit.build_fold_manifest(METADATA, CONFIG)
SELECTED, SELECTION_SUMMARY, _BIN_EDGES = audit.select_distribution_matched_cutoffs(
    METADATA, CANDIDATES, CONFIG
)
REPORT = audit.distribution_report(METADATA, CANDIDATES, SELECTED, CONFIG)
REPLAY = audit.build_replay_requests(SELECTED, METADATA, FOLD_MANIFEST, CONFIG)
LEAKAGE = audit.assert_leakage_contract(REPLAY, FOLD_MANIFEST, CONFIG)
GUARD = audit.evaluate_distribution_guard(
    REPORT, SELECTION_SUMMARY, LEAKAGE, CONFIG
)
if not GUARD["pass"]:
    raise AssertionError("distribution/leakage guard failed before cache generation")
print(
    f"replay_requests={len(REPLAY)} selected_wells={REPLAY['source_well'].nunique()} "
    f"distribution_guard={GUARD['pass']}"
)


# %% [markdown]
# ## 4. Prefix materialization
#
# True TVT after each synthetic cutoff is kept target-only. Feature builders
# receive the masked prefix surface and inherit the source-well fold.

# %%
MATERIALIZED, REQUEST_SUMMARY, MATERIALIZATION_SCHEMA, MATERIALIZATION_AUDIT = (
    audit.materialize_prefix_features(REPLAY, FRAMES, CONFIG)
)
if not MATERIALIZATION_AUDIT["pass"]:
    raise AssertionError("prefix materialization guard failed")
print(
    f"materialized_requests={MATERIALIZED['request_id'].nunique()} "
    f"materialized_rows={len(MATERIALIZED)}"
)


# %% [markdown]
# ## 5. Chunked exp218 feature-cache generation
#
# Requests are processed in deterministic sorted batches. Each completed batch
# is written as one Parquet shard and released before the next batch starts.

# %%
CACHE_SUMMARY = augmentation.run_chunked_feature_cache_generation(
    replay=REPLAY,
    materialized=MATERIALIZED,
    frames=FRAMES,
    raw_train_dir=TRAIN_DIR,
    config=CONFIG,
    output_dir=audit.artifact_dir(),
)


# %% [markdown]
# ## 6. Cache manifest and reproducibility summary

# %%
print(CACHE_SUMMARY)
