# %% [markdown]
# # exp266 PF ANCC / PF-Z multiseed stability audit inference

# %% [markdown]
# ## Contents
# 1. Policy
# 2. Disabled inference guard

# %%
from __future__ import annotations

EXPERIMENT_NAME = "exp266_pf_ancc_pf_z_multiseed_stability_audit"

# %% [markdown]
# ## 1. Policy
#
# This experiment is a train-side stochastic stability audit. It does not select
# a deployable candidate and must not create a submission.

# %%
POLICY = {
    "route": "pf_beam",
    "inference_enabled": False,
    "submission_enabled": False,
    "reason": "train-side multiseed diagnostic only",
}
print(EXPERIMENT_NAME, POLICY)

# %% [markdown]
# ## 2. Disabled inference guard

# %%
if POLICY["inference_enabled"] or POLICY["submission_enabled"]:
    raise RuntimeError("exp266 inference and submission must remain disabled")
print("Inference is intentionally disabled; no submission.csv is created.")
