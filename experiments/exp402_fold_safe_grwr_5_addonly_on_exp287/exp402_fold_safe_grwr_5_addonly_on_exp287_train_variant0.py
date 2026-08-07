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
# # exp402 Stage 0A — train source components and outer roles
#
# version 1のmonolithic preflightからtrain-sideだけを分離する。
# exp072 train cache、exp287 fold-safe formation roles、exp264 OOFをSHA固定で読み、
# source component 3列とouter-fold×roleのGRWR-5 10 partitionsを生成する。
# model、booster、TVT prediction、submissionは生成しない。

# %% [markdown]
# ## 1. Imports and split phase

# %%
import os

os.environ["EXP402_IMPORT_ONLY"] = "1"

from IPython.display import display

import exp402_fold_safe_grwr_5_addonly_on_exp287_compact_selfcontained_train as exp402


PHASE = "train_roles"

# %% [markdown]
# ## 2. Configuration and cost contract

# %%
CONFIG = exp402.read_yaml(exp402.find_config_path())
CONTRACT = exp402.validate_scientific_contract(
    CONFIG,
    require_run_approval=True,
)
display(
    {
        "experiment": exp402.EXPERIMENT_NAME,
        "phase": PHASE,
        "route": CONFIG["experiment"]["route"],
        "parent": CONFIG["lineage"]["parent"],
        "input_kernels": CONFIG["runtime"]["kaggle"][
            "train_variant0_kernel_sources"
        ],
        "models": CONTRACT["current_execution"]["models"],
        "boosters": CONTRACT["current_execution"]["boosters"],
        "predictions": CONTRACT["current_execution"]["predictions"],
        "submissions": CONTRACT["current_execution"]["submissions"],
    }
)

# %% [markdown]
# ## 3. Generate train-side Stage 0 artifacts

# %%
RESULT = exp402.run_experiment(CONFIG, phase=PHASE)
display(RESULT)

# %% [markdown]
# ## 4. Generated files

# %%
ARTIFACT_ROOT = exp402.KAGGLE_WORKING_ROOT / "artifacts"
for path in sorted(ARTIFACT_ROOT.rglob("*")):
    if path.is_file():
        print(path.relative_to(ARTIFACT_ROOT), path.stat().st_size)
