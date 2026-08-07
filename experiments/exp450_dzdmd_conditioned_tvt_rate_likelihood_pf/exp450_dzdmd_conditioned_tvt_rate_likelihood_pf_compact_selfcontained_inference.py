# %% [markdown]
# # exp450 dZ/dMD-conditioned TVT-rate likelihood-PF — inference guard
#
# Raw-test generation and submission are outside the approved exp450
# implementation scope. This notebook is an explicit fail-closed guard, not an
# inference implementation.

# %% [markdown]
# ## Contents
# 1. Imports
# 2. Frozen authorization guard

# %%
from __future__ import annotations

import os


EXPERIMENT_NAME = "exp450_dzdmd_conditioned_tvt_rate_likelihood_pf"


def in_notebook_runtime() -> bool:
    try:
        return get_ipython() is not None  # type: ignore[name-defined]
    except NameError:
        return False


EXECUTE_NOTEBOOK = os.environ.get("EXP450_IMPORT_ONLY", "0") != "1" and in_notebook_runtime()


# %% [markdown]
# ## 2. Frozen authorization guard


# %%
def run_inference() -> None:
    raise RuntimeError(
        "exp450 raw-test inference/submission is not implemented or approved; "
        "Stage 1 must first pass under a separate user approval"
    )


if EXECUTE_NOTEBOOK:
    run_inference()
