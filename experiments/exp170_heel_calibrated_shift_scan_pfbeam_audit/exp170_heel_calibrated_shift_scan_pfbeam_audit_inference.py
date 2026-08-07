# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---
# %% [markdown]
# # exp170_heel_calibrated_shift_scan_pfbeam_audit inference

# %% [markdown]
# ## Contents
# 1. Imports
# 2. Runtime and configuration
# 3. Inference policy

# %% [markdown]
# ## 1. Imports

# %%
from __future__ import annotations

from IPython.display import display
from settings import ExperimentPaths, get_nested, load_config

# %% [markdown]
# ## 2. Runtime and configuration

# %%
paths = ExperimentPaths()
config = load_config()
paths.require_kaggle_runtime()
paths.ensure_output_dirs()

display(
    {
        "experiment": get_nested(config, "experiment.name"),
        "route": get_nested(config, "experiment.route"),
        "inference_mode": get_nested(config, "inference.mode"),
        "selected_candidate": get_nested(config, "inference.selected_candidate"),
        "train_dir": str(paths.train_data_dir),
        "artifacts_dir": str(paths.artifacts_dir),
    }
)

# %% [markdown]
# ## 3. Inference policy

# %%
raise RuntimeError(
    "exp170 is a train-side diagnostic audit only. "
    "No inference port, candidate replacement, or submission is selected."
)
