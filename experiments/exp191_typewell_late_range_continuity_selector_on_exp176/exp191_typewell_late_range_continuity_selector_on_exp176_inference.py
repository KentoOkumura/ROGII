# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---
# %% [markdown]
# # exp191_typewell_late_range_continuity_selector_on_exp176 inference

# %% [markdown]
# ## Contents
# 1. Imports
# 2. Configuration
# 3. Inference status

# %%
from __future__ import annotations

import json

from settings import get_nested, load_config

# %% [markdown]
# ## 2. Configuration

# %%
config = load_config()
print("experiment:", get_nested(config, "experiment.name"))
print("route:", get_nested(config, "experiment.route"))
print("status:", get_nested(config, "experiment.status"))
print("parent:", get_nested(config, "lineage.parent"))
print("selected_variant:", get_nested(config, "inference.selected_variant"))

# %% [markdown]
# ## 3. Inference status

# %%
notes = get_nested(config, "inference.notes") or []
print(json.dumps({"mode": get_nested(config, "inference.mode"), "notes": notes}, indent=2))
print("submission_generation:", False)
print("direct_replacement:", False)
