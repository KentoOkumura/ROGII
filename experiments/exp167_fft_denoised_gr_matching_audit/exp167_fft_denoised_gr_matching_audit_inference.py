# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---
# %% [markdown]
# # exp167_fft_denoised_gr_matching_audit inference

# %% [markdown]
# ## Contents
# 1. Imports
# 2. Runtime and configuration
# 3. Inference status

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
        "mode": get_nested(config, "audit.mode"),
    }
)

# %% [markdown]
# ## 3. Inference status

# %%
print(
    "This experiment is a train-side FFT-denoised GR matching audit. "
    "It does not generate inference predictions or submission.csv."
)
