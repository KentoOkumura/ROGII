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
# # exp402 Stage 0F — outer fold 2

# %%
import os

os.environ["EXP402_IMPORT_ONLY"] = "1"

from IPython.display import display

import exp402_fold_safe_grwr_5_addonly_on_exp287_compact_selfcontained_train as exp402


PHASE = "train_fold"
OUTER_FOLD = 2
CONFIG = exp402.read_yaml(exp402.find_config_path())
CONTRACT = exp402.validate_scientific_contract(CONFIG, require_run_approval=True)
display(
    {
        "experiment": exp402.EXPERIMENT_NAME,
        "phase": PHASE,
        "outer_fold": OUTER_FOLD,
        "models": CONTRACT["current_execution"]["models"],
        "boosters": CONTRACT["current_execution"]["boosters"],
        "formation_partition_reads": 2,
    }
)

# %%
RESULT = exp402.run_experiment(CONFIG, phase=PHASE, outer_fold=OUTER_FOLD)
display(RESULT)

# %%
ARTIFACT_ROOT = exp402.KAGGLE_WORKING_ROOT / "artifacts"
for path in sorted(ARTIFACT_ROOT.rglob("*")):
    if path.is_file():
        print(path.relative_to(ARTIFACT_ROOT), path.stat().st_size)
