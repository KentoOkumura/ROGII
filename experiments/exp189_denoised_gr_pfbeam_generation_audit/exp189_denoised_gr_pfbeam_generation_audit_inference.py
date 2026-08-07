# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---
# %% [markdown]
# # exp189_denoised_gr_pfbeam_generation_audit inference
#
# This experiment is a train-side PF/Beam generation audit. It intentionally does
# not create inference predictions or a submission candidate.

# %% [markdown]
# ## Contents
# 1. Imports
# 2. Configuration check
# 3. Inference guard

# %% [markdown]
# ## 1. Imports

# %%
from __future__ import annotations

from IPython.display import display
from settings import ExperimentPaths, get_nested, load_config

# %% [markdown]
# ## 2. Configuration check

# %%
paths = ExperimentPaths()
config = load_config()
paths.require_kaggle_runtime()

display(
    {
        "experiment": get_nested(config, "experiment.name"),
        "route": get_nested(config, "experiment.route"),
        "inference_mode": get_nested(config, "inference.mode"),
        "selected_candidate": get_nested(config, "inference.selected_candidate"),
        "notes": get_nested(config, "inference.notes"),
    }
)

# %% [markdown]
# ## 3. Inference guard

# %%
raise RuntimeError(
    "exp189 is train-side audit only. Do not run inference or create submission.csv from this exp."
)
