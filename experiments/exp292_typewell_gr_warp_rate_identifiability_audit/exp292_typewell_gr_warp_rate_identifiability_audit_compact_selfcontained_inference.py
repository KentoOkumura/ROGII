# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # exp292 inference — intentionally disabled
#
# exp292 is a train-side identifiability audit. It does not define a deployable
# decoder, persist selected row predictions, read raw test wells, or create a
# submission. A future safe-preserving inference policy requires a separate
# steering document and explicit approval even if the audit passes.

# %%
from __future__ import annotations

import json

CONTRACT = {
    "experiment": "exp292_typewell_gr_warp_rate_identifiability_audit",
    "route": "pf_beam",
    "inference_enabled": False,
    "create_submission": False,
    "reason": "train-side identifiability audit only; decoder requires separate approval",
}

print(json.dumps(CONTRACT, indent=2, sort_keys=True))
raise RuntimeError(CONTRACT["reason"])
