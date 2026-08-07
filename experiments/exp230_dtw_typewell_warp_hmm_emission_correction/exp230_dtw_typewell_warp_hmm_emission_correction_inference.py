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
# # exp230 dtw typewell warp hmm emission correction inference

# %% [markdown]
# ## Contents
# 1. Imports
# 2. Configuration check
# 3. Inference policy

# %%
from __future__ import annotations

import json

from exact_hmm_smoother import to_jsonable
from settings import ExperimentPaths, get_nested, load_config


# %% [markdown]
# ## 2. Configuration check

# %%
paths = ExperimentPaths()
config = load_config()
inference = get_nested(config, "inference") or {}
payload = {
    "experiment": paths.experiment_name,
    "route": get_nested(config, "experiment.route"),
    "inference_mode": inference.get("mode"),
    "selected_candidate": inference.get("selected_candidate"),
    "notes": inference.get("notes"),
}
print(json.dumps(to_jsonable(payload), indent=2, sort_keys=True))


# %% [markdown]
# ## 3. Inference policy

# %%
if inference.get("mode") != "not_applicable_train_feature_cache_only":
    raise ValueError("exp230 inference notebook is only valid for train-feature-cache-only mode")

print(
    "exp230 intentionally does not generate raw-test features, predictions, "
    "submission.csv, or Kaggle code submissions. Run the train notebook only."
)
