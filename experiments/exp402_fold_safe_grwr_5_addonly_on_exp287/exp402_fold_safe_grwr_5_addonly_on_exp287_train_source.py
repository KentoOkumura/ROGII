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
# # exp402 Stage 0A — train context/source
#
# 親OOFとclean候補を整合し、GR/DWT/FFT source 3列を一度だけ生成する。
# 巨大なformation role parquetはこのrunでは読まない。

# %%
import os

os.environ["EXP402_IMPORT_ONLY"] = "1"

from IPython.display import display

import exp402_fold_safe_grwr_5_addonly_on_exp287_compact_selfcontained_train as exp402


PHASE = "train_source"
CONFIG = exp402.read_yaml(exp402.find_config_path())
CONTRACT = exp402.validate_scientific_contract(CONFIG, require_run_approval=True)
display(
    {
        "experiment": exp402.EXPERIMENT_NAME,
        "phase": PHASE,
        "route": CONFIG["experiment"]["route"],
        "models": CONTRACT["current_execution"]["models"],
        "boosters": CONTRACT["current_execution"]["boosters"],
        "formation_partition_reads": 0,
    }
)

# %%
RESULT = exp402.run_experiment(CONFIG, phase=PHASE)
display(RESULT)

# %%
ARTIFACT_ROOT = exp402.KAGGLE_WORKING_ROOT / "artifacts"
for path in sorted(ARTIFACT_ROOT.rglob("*")):
    if path.is_file():
        print(path.relative_to(ARTIFACT_ROOT), path.stat().st_size)
