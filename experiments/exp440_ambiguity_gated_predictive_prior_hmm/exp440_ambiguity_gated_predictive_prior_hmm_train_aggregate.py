# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---
# %% [markdown]
# # exp440 full OOF — strict four-shard merge and truth-late readout

# %%
import os

os.environ["EXP440_STAGE"] = "stage1_merge"

import exp440_ambiguity_gated_predictive_prior_hmm_compact_selfcontained_train as implementation  # noqa: E402

CONFIG = implementation.load_config()
SUMMARY = implementation.run_selected_stage(CONFIG)
