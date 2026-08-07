# %% [markdown]
# # exp362 segment local donor slope exact HMM inference

# %% [markdown]
# ## Contents
# 1. Imports
# 2. Frozen inference policy
# 3. Fail-closed execution guard

# %% [markdown]
# ## 1. Imports

# %%
from __future__ import annotations

import os

EXPERIMENT_NAME = "exp362_segment_local_donor_slope_exact_hmm"
EXECUTE_NOTEBOOK = os.environ.get("EXP362_IMPORT_ONLY") != "1"


# %% [markdown]
# ## 2. Frozen inference policy

# %%
INFERENCE_POLICY = {
    "experiment": EXPERIMENT_NAME,
    "enabled": False,
    "create_submission": False,
    "reason": (
        "The design-frozen scope is one train-side exact-HMM scientific run. "
        "Inference requires a passed CV gate and a separate user decision."
    ),
    "forbidden_inputs": [
        "exp226_oof",
        "tvt_geop",
        "tvt_pred",
        "gr_delta",
        "adaptive_kappa",
        "near_strike_ancc",
        "exp226_u_projection",
    ],
}
print("Experiment:", EXPERIMENT_NAME)
print("Inference enabled:", INFERENCE_POLICY["enabled"])
print("Create submission:", INFERENCE_POLICY["create_submission"])


# %% [markdown]
# ## 3. Fail-closed execution guard


# %%
def assert_inference_disabled() -> None:
    raise RuntimeError(
        "exp362 inference and submission are disabled. "
        "Do not create a sample-submission copy or raw-test prediction."
    )


if EXECUTE_NOTEBOOK:
    assert_inference_disabled()
