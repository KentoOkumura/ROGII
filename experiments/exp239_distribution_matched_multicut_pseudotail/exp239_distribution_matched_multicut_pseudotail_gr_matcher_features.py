# %% [markdown]
# # exp239 official exp218 feature cache (CPU)
#
# Build the official-start 380-feature exp218 surface once on CPU and persist
# it as bounded Parquet shards. This stage trains no model and uses no GPU.

# %% [markdown]
# ## Contents
# 1. Imports and runtime helpers
# 2. Configuration and official input contract
# 3. exp072 and exp145 cache loading
# 4. U projection and GRWR feature assembly
# 5. Row-sharded Parquet cache generation
# 6. Schema, SHA, and memory summary

# %%
from __future__ import annotations

import importlib
import os
from pathlib import Path

os.environ["EXP239_IMPORT_ONLY"] = "1"

audit = importlib.import_module("exp239_distribution_matched_multicut_pseudotail_train")
augmentation = importlib.import_module("exp239_exp218_pseudotail_augmentation")


# %% [markdown]
# ## 2. Configuration and official input contract
#
# Expected output is the unchanged exp218 official-start surface: 3,783,989
# rows, 773 wells, and 380 model features. No pseudo rows are loaded here.

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
print(f"config={CONFIG_PATH}")
print(f"route={audit.nested(CONFIG, 'experiment.route')}")
print("stage=cpu_official_feature_cache")
print(f"train_dir={TRAIN_DIR}")
print(
    "expected="
    f"rows={audit.nested(CONFIG, 'model.exp218_augmentation.expected_official_rows')} "
    f"features={audit.nested(CONFIG, 'model.exp218_augmentation.expected_feature_count')} "
    "row_batch="
    f"{audit.nested(CONFIG, 'model.exp218_augmentation.official_feature_cache.row_batch_count')} "
    "configs=0 folds=0 boosters=0 control_retrained=false"
)
if audit.nested(CONFIG, "model.exp218_augmentation.execution_stage") != (
    "cpu_official_feature_cache"
):
    raise AssertionError("config execution_stage must be cpu_official_feature_cache")


# %% [markdown]
# ## 3-5. Official feature assembly and row-sharded cache generation
#
# exp072 base and exp145 learned-likelihood caches are joined with the same U
# projection and target-free GRWR builders used by exp218. The completed frame
# is written in 250,000-row shards to avoid another full-frame training copy.

# %%
OFFICIAL_CACHE_SUMMARY = augmentation.run_official_feature_cache_generation(
    raw_train_dir=TRAIN_DIR,
    config=CONFIG,
    output_dir=audit.artifact_dir(),
)


# %% [markdown]
# ## 6. Schema, SHA, and memory summary
#
# Every shard records file SHA and deterministic row-content SHA. The summary
# also records the 380-column schema SHA, total rows, wells, and peak RSS.

# %%
print(OFFICIAL_CACHE_SUMMARY)
