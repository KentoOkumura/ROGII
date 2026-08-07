# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---
# %% [markdown]
# # exp243 PF seed medoids — shard 1
#
# Canonical exp243 train notebookをdeterministic well shard 1で実行するentrypoint。

# %%
import os

os.environ["EXP243_ACTIVE_WELL_SHARD_INDEX"] = "1"

import exp243_pf_seed_medoids_train  # noqa: F401, E402
