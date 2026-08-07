# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---
# %% [markdown]
# # exp410 PF mechanism counterfactual — one-well preflight
#
# Runs all 12 preregistered paired variants on the first fixed shard-0 sentinel
# to validate Numba compilation, parity, runtime, memory, and artifact schema.

# %%
import os

os.environ["EXP410_COUNTERFACTUAL_SHARD_INDEX"] = "0"
os.environ["EXP410_COUNTERFACTUAL_PREFLIGHT"] = "1"

import run_counterfactual_sentinels  # noqa: E402,F401
