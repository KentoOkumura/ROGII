# %% [markdown]
# # exp268 multi-scale initial-rate candidates inference
#
# This experiment is a train-side exact-HMM candidate-bank audit. Raw-test
# regeneration, candidate selection, submission creation, and Kaggle submission
# are intentionally disabled.

# %% [markdown]
# ## Contents
# 1. Immutable inference contract
# 2. Disabled inference guard
# 3. Expected continuation after train-side review

# %% [markdown]
# ## 1. Immutable inference contract

# %%
from __future__ import annotations

import json

CONTRACT = {
    "experiment": "exp268_multi_scale_initial_rate_candidates",
    "route": "pf_beam",
    "status": "train_side_candidate_bank_audit_only",
    "raw_test_regeneration": False,
    "candidate_selection": False,
    "candidate_mean": False,
    "oracle_selection": False,
    "submission_creation": False,
}
print(json.dumps(CONTRACT, indent=2, sort_keys=True))


# %% [markdown]
# ## 2. Disabled inference guard

# %%
raise RuntimeError(
    "exp268 is train-side only. Do not regenerate raw-test HMM candidates, "
    "select a rate window, average candidates, or create submission.csv before "
    "the row/block/whole-well headroom and target-free selectability are reviewed."
)


# %% [markdown]
# ## 3. Expected continuation after train-side review
#
# If the frozen candidate bank shows material block/whole-well headroom, any
# target-free selector or raw-test regeneration must be designed explicitly in
# this same experiment before inference is enabled. A positive row oracle alone
# is not sufficient.
