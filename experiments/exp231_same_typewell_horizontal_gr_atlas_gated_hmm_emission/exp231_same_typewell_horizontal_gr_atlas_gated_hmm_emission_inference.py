# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
# kernelspec:
#   display_name: Python 3
#   language: python
#   name: python3
# ---

# %% [markdown]
# # exp231 same-typewell horizontal GR atlas gated HMM emission — inference policy

# %% [markdown]
# ## Contents
# 1. Imports
# 2. Configuration and raw-test guard
# 3. Inference policy

# %%
from __future__ import annotations

import json

from exact_hmm_smoother import to_jsonable
from settings import ExperimentPaths, get_nested, load_config


# %% [markdown]
# ## 2. Configuration and raw-test guard

# %%
paths = ExperimentPaths()
config = load_config()
inference = get_nested(config, "inference") or {}
peer_atlas = get_nested(config, "model.peer_atlas_emission") or {}
payload = {
    "experiment": paths.experiment_name,
    "route": get_nested(config, "experiment.route"),
    "inference_mode": inference.get("mode"),
    "selected_candidate": inference.get("selected_candidate"),
    "peer_atlas_enabled": peer_atlas.get("enabled"),
    "raw_test_peer_policy": "not approved",
    "notes": inference.get("notes"),
}
print(json.dumps(to_jsonable(payload), indent=2, sort_keys=True))


# %% [markdown]
# ## 3. Inference policy

# %%
if inference.get("mode") != "not_applicable_train_feature_cache_only":
    raise ValueError("exp231 inference is disabled until the fold-safe atlas passes train-side guards")

print(
    "exp231 intentionally creates no raw-test atlas, prediction, submission.csv, or code submission. "
    "A follow-up in this same experiment requires user confirmation after the train-side readout."
)
