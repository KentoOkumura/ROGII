# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---
# %% [markdown]
# # exp243 PF seed medoids — inference disabled
#
# この実験はtrain-side candidate audit専用である。selectorとsafety guardが成立するまで、
# raw-test PF再生成、inference、submissionを行わない。

# %% [markdown]
# ## Contents
# 1. Configuration
# 2. Disabled inference contract

# %% [markdown]
# ## 1. Configuration

# %%
from IPython.display import display

from settings import get_nested, load_config

config = load_config()
display(
    {
        "experiment": get_nested(config, "experiment.name"),
        "route": get_nested(config, "experiment.route"),
        "inference": get_nested(config, "inference"),
    }
)

# %% [markdown]
# ## 2. Disabled inference contract

# %%
if get_nested(config, "inference.mode") != "disabled_diagnostic_only":
    raise RuntimeError("exp243 inference must remain disabled")

print(
    "No raw-test regeneration or submission was produced. "
    "Run the train-side medoid candidate audit and establish a safe selector first."
)
