# %% [markdown]
# # exp273 two-dimensional formation-gradient transition inference
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
    "experiment": "exp273_two_dimensional_formation_gradient_transition",
    "route": "pf_beam",
    "status": "train_side_candidate_bank_audit_only",
    "raw_test_regeneration": False,
    "candidate_selection": False,
    "candidate_mean": False,
    "gradient_hard_switch": False,
    "direct_tvt_correction": False,
    "oracle_selection": False,
    "submission_creation": False,
}
print(json.dumps(CONTRACT, indent=2, sort_keys=True))


# %% [markdown]
# ## 2. Disabled inference guard

# %%
raise RuntimeError(
    "exp273 is train-side only. Do not fit a test gradient, regenerate raw-test "
    "HMM candidates, hard-switch/average candidates, directly correct TVT, or "
    "create submission.csv before geometry guards, block/whole-well headroom, "
    "hidden-like behavior, and worst-well safety are reviewed."
)


# %% [markdown]
# ## 3. Expected continuation after train-side review
#
# If the frozen candidate bank shows material block/whole-well headroom, any
# target-free selector or raw-test regeneration must be designed explicitly in
# this same experiment before inference is enabled. A positive row oracle or a
# turning-well-only gain is not sufficient.
