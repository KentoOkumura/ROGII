# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---
# %% [markdown]
# # exp187_cluster_outlier_alt_typewell_pfbeam_audit inference

# %% [markdown]
# ## Contents
# 1. Imports
# 2. Runtime and configuration check
# 3. Inference status

# %% [markdown]
# ## 1. Imports

# %%
from __future__ import annotations

from IPython.display import display
from settings import ExperimentPaths, get_nested, load_config

# %% [markdown]
# ## 2. Runtime and configuration check

# %%
paths = ExperimentPaths()
config = load_config()
paths.require_kaggle_runtime()
paths.ensure_output_dirs()

display(
    {
        "experiment": get_nested(config, "experiment.name"),
        "route": get_nested(config, "experiment.route"),
        "mode": get_nested(config, "inference.mode"),
        "selected_candidate": get_nested(config, "inference.selected_candidate"),
        "notes": get_nested(config, "inference.notes"),
    }
)

# %% [markdown]
# ## 3. Inference status

# %%
raise RuntimeError(
    "exp187 is a train-side PF/Beam audit only. "
    "It intentionally does not create an inference port or submission.csv."
)
