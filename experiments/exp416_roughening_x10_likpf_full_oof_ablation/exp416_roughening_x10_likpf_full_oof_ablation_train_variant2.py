# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---
# %% [markdown]
# # exp416 roughening x10 likelihood-PF — shard 2

# %%
import os

os.environ["EXP416_STAGE"] = "shard"
os.environ["EXP416_SHARD_INDEX"] = "2"

import exp416_roughening_x10_likpf_full_oof_ablation_compact_selfcontained_train  # noqa: E402,F401
