# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---
# %% [markdown]
# # exp241 containment audit — shard 0
#
# Canonical exp241 train notebookをwell shard 0で実行するKaggle package entrypoint。

# %%
import os

os.environ["EXP241_ACTIVE_WELL_SHARD_INDEX"] = "0"

import exp241_adaptive_likelihood_pf_trajectory_containment_audit_train  # noqa: F401, E402
