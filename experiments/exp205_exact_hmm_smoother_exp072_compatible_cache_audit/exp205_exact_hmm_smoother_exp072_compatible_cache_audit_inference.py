# %% [markdown]
# # exp205 exact HMM smoother exp072 compatible cache audit inference

# %% [markdown]
# ## Contents
# 1. Imports
# 2. Setup and configuration
# 3. Not applicable inference guard
# 4. Expected next step

# %%
from __future__ import annotations

import json

from exact_hmm_smoother import to_jsonable
from settings import ExperimentPaths, get_nested, load_config


# %% [markdown]
# ## 2. Setup and configuration

# %%
paths = ExperimentPaths()
config = load_config()
payload = {
    "experiment": get_nested(config, "experiment.name"),
    "route": get_nested(config, "experiment.route"),
    "inference_mode": get_nested(config, "inference.mode"),
    "artifacts_dir": str(paths.artifacts_dir),
}
print(json.dumps(to_jsonable(payload), indent=2, sort_keys=True))


# %% [markdown]
# ## 3. Not applicable inference guard

# %%
if get_nested(config, "inference.mode") != "not_applicable_train_feature_cache_only":
    raise ValueError("exp205 inference notebook is only valid for train-feature-cache-only mode")

print(
    "exp205 intentionally does not generate raw-test HMM features, predictions, "
    "submission.csv, or Kaggle competition submissions."
)


# %% [markdown]
# ## 4. Expected next step

# %%
print(
    "Use the train notebook outputs for train-side direct comparison. "
    "If supported, create a later experiment for raw-test-compatible HMM regeneration."
)
