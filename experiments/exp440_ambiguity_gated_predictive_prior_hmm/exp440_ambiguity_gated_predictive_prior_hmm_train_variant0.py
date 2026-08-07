# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---
# %% [markdown]
# # exp440 full OOF — deterministic LPT shard 0

# %%
import os

os.environ["EXP440_STAGE"] = "stage1_shard"
os.environ["EXP440_SHARD_INDEX"] = "0"

import exp440_ambiguity_gated_predictive_prior_hmm_compact_selfcontained_train as implementation  # noqa: E402

CONFIG = implementation.load_config()
SUMMARY = implementation.run_selected_stage(CONFIG)
