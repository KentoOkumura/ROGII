# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---
# %% [markdown]
# # exp177_beam_topk_bimodal_gate_posthoc_audit inference

# %% [markdown]
# ## Contents
# 1. Imports
# 2. Configuration
# 3. No-submission guard

# %%
from __future__ import annotations

from settings import get_nested, load_config

# %% [markdown]
# ## 2. Configuration

# %%
config = load_config()
print("experiment:", get_nested(config, "experiment.name"))
print("route:", get_nested(config, "experiment.route"))
print("status:", get_nested(config, "experiment.status"))
print("parent:", get_nested(config, "lineage.parent"))

# %% [markdown]
# ## 3. No-submission guard

# %%
raise RuntimeError(
    "exp177 is a train-side posthoc audit only. It does not define an inference "
    "candidate or submission.csv. Run the train notebook for diagnostics."
)
