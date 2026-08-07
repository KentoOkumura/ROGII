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
# # exp402 Stage 0B — raw current-test GRWR-5 replay
#
# competition raw train/testとSHA固定したexp072 replay sourceだけを使い、
# current-test 3 wellsのPF/Beam候補、formation surface、GRWR-5を再生成する。
# Stage 0Aのtrain-side生成物は読まず、model、booster、TVT prediction、
# submissionも生成しない。

# %% [markdown]
# ## 1. Imports and split phase

# %%
import os

os.environ["EXP402_IMPORT_ONLY"] = "1"

from IPython.display import display

import exp402_fold_safe_grwr_5_addonly_on_exp287_compact_selfcontained_train as exp402


PHASE = "current_test"

# %% [markdown]
# ## 2. Configuration and fixed replay cost

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
        "test_wells": CONTRACT["stage_0_current_test_regeneration"][
            "test_wells"
        ],
        "pf_ancc_well_runs": CONTRACT["stage_0_current_test_regeneration"][
            "pf_ancc_well_runs"
        ],
        "pf_z_well_runs": CONTRACT["stage_0_current_test_regeneration"][
            "pf_z_well_runs"
        ],
        "beam_paths": CONTRACT["stage_0_current_test_regeneration"][
            "beam_paths"
        ],
        "likelihood_pf_particle_starts": CONTRACT[
            "stage_0_current_test_regeneration"
        ]["likelihood_pf_particle_starts"],
        "models": CONTRACT["current_execution"]["models"],
        "boosters": CONTRACT["current_execution"]["boosters"],
        "predictions": CONTRACT["current_execution"]["predictions"],
        "submissions": CONTRACT["current_execution"]["submissions"],
    }
)

# %% [markdown]
# ## 3. Generate current-test Stage 0 artifacts

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
