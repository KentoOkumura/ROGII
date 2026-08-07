# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---
# %% [markdown]
# # exp410 PF mechanism counterfactual — shard 0
#
# Runs the preregistered initialization, transition, GR, resampling, and
# roughening interventions on the fixed sentinel wells assigned to shard 0.

# %%
import os

os.environ["EXP410_COUNTERFACTUAL_SHARD_INDEX"] = "0"

import run_counterfactual_sentinels  # noqa: E402,F401
