# %% [markdown]
# # exp239 pseudo-tail dual-cache streaming augmentation training (GPU)
#
# Train the approved exp218 augmentation variant from validated official and
# pseudo-tail CPU feature caches. The saved exp218 OOF is the control; this
# notebook does not retrain it.

# %% [markdown]
# ## Contents
# 1. Imports and runtime helpers
# 2. Configuration and dual-cache input contract
# 3. Shard validation and disk-backed array streaming
# 4. Fold-safe cached pseudo-tail injection
# 5. Three-config, five-fold LightGBM training
# 6. Official-start OOF metrics and generated artifacts

# %%
from __future__ import annotations

import importlib
import os
from pathlib import Path

os.environ["EXP239_IMPORT_ONLY"] = "1"

audit = importlib.import_module("exp239_distribution_matched_multicut_pseudotail_train")
augmentation = importlib.import_module("exp239_exp218_pseudotail_augmentation")


# %% [markdown]
# ## 2. Configuration and cache input contract
#
# Required totals are 16 official shards / 3,783,989 rows and 32 pseudo shards /
# 800 requests / 799,961 rows, with a shared 380-feature schema. Every shard
# file SHA and deterministic row-content SHA is checked while streaming.

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
print("stage=gpu_dual_cache_streaming_training")
print(f"train_dir={TRAIN_DIR}")
print(
    "approved_training="
    f"variant={audit.nested(CONFIG, 'model.exp218_augmentation.variant')} "
    f"official_weight={audit.nested(CONFIG, 'model.exp218_augmentation.official_row_weight')} "
    f"pseudo_weight={audit.nested(CONFIG, 'model.exp218_augmentation.pseudo_row_weight')} "
    f"configs={audit.nested(CONFIG, 'model.exp218_augmentation.lightgbm_configs')} "
    f"folds={audit.nested(CONFIG, 'model.exp218_augmentation.folds')} "
    f"boosters={audit.nested(CONFIG, 'model.exp218_augmentation.boosters')} "
    "control_retrained="
    f"{audit.nested(CONFIG, 'model.exp218_augmentation.parent_control_retrained')}"
)
if (
    audit.nested(CONFIG, "model.exp218_augmentation.execution_stage")
    != "gpu_dual_cache_streaming_training"
):
    raise AssertionError("config execution_stage must be gpu_dual_cache_streaming_training")


# %% [markdown]
# ## 3-5. Dual-cache validation, memmap streaming, and GPU training
#
# Validation rows are official-start rows only. For every outer fold, pseudo
# rows derived from any validation source well are excluded from training.
# Both caches and fold matrices are disk-backed. Fold matrices are populated
# with `numpy.take` and deleted after each booster; no giant `vstack` or full
# cache DataFrame is created.

# %%
TRAINING_SUMMARY = augmentation.run_cached_augmentation_evaluation(
    raw_train_dir=TRAIN_DIR,
    config=CONFIG,
    output_dir=audit.artifact_dir(),
)


# %% [markdown]
# ## 6. Official-start OOF metrics and generated artifacts
#
# The main decision statistic is delta against saved exp218 OOF RMSE
# 8.475793752. Models, per-fold metrics, feature importance, OOF predictions,
# schema SHA, cache manifest SHA, and model SHA values are written to artifacts.

# %%
print(TRAINING_SUMMARY)
