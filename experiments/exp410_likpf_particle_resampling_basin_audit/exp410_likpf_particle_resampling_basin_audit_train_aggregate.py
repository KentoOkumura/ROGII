# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---
# %% [markdown]
# # exp410 PF particle audit — fixed four-well preflight

# %%
import os

os.environ["EXP410_RUN_STAGE"] = "preflight"
os.environ["EXP410_ACTIVE_WELL_SHARD_INDEX"] = "0"

import exp410_likpf_particle_resampling_basin_audit_compact_selfcontained_train  # noqa: E402,F401
