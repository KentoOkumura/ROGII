# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---
# %% [markdown]
# # exp402 Stage 0C — split preflight aggregate
#
# Stage 0A、outer-fold 5 shard、Stage 0Bのimmutable Kaggle Notebook outputsを
# 入力にし、source/config SHA、10 role partitionのfile/content ledger、
# current-test SHA、leakage/cost guardを一つのpreflight manifestへ統合する。

# %% [markdown]
# ## 1. Imports and split phase

# %%
import os

os.environ["EXP402_IMPORT_ONLY"] = "1"

from IPython.display import display

import exp402_fold_safe_grwr_5_addonly_on_exp287_compact_selfcontained_train as exp402


PHASE = "aggregate"
FOLD4_RUNTIME_ARTIFACT_ROOT = (
    "/kaggle/input/notebooks/"
    "kentookumura/exp402-foldsafe-grwr5-train-fold4-v2/artifacts"
)

# %% [markdown]
# ## 2. Configuration and upstream outputs

# %%
CONFIG = exp402.read_yaml(exp402.find_config_path())
CONTRACT = exp402.validate_scientific_contract(
    CONFIG,
    require_run_approval=True,
)
SPLIT = CONFIG["runtime"]["kaggle"]["split_stage_0"]
fold4_artifact_patterns = SPLIT["train_folds"]["4"]["artifact_root_patterns"]
if FOLD4_RUNTIME_ARTIFACT_ROOT not in fold4_artifact_patterns:
    fold4_artifact_patterns.insert(0, FOLD4_RUNTIME_ARTIFACT_ROOT)
display(
    {
        "experiment": exp402.EXPERIMENT_NAME,
        "phase": PHASE,
        "route": CONFIG["experiment"]["route"],
        "train_source_kernel": SPLIT["train_source"]["kernel_id"],
        "train_fold_kernels": [
            SPLIT["train_folds"][str(outer_fold)]["kernel_id"]
            for outer_fold in range(5)
        ],
        "current_test_kernel": SPLIT["current_test"]["kernel_id"],
        "fold4_runtime_artifact_root": FOLD4_RUNTIME_ARTIFACT_ROOT,
        "models": CONTRACT["current_execution"]["models"],
        "boosters": CONTRACT["current_execution"]["boosters"],
        "predictions": CONTRACT["current_execution"]["predictions"],
        "submissions": CONTRACT["current_execution"]["submissions"],
    }
)

# %% [markdown]
# ## 3. Verify and aggregate Stage 0 evidence

# %%
RESULT = exp402.run_experiment(CONFIG, phase=PHASE)
display(RESULT)

# %% [markdown]
# ## 4. Generated manifests

# %%
ARTIFACT_ROOT = exp402.KAGGLE_WORKING_ROOT / "artifacts"
for path in sorted(ARTIFACT_ROOT.rglob("*")):
    if path.is_file():
        print(path.relative_to(ARTIFACT_ROOT), path.stat().st_size)
