# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---
# %% [markdown]
# # exp416 roughening x10 likelihood-PF — strict merge and readout

# %%
import os

os.environ["EXP416_STAGE"] = "merge"

import exp416_roughening_x10_likpf_full_oof_ablation_compact_selfcontained_train  # noqa: E402,F401
