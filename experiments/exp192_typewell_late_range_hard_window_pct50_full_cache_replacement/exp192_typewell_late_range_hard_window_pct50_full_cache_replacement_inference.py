# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---
# %% [markdown]
# # exp192_typewell_late_range_hard_window_pct50_full_cache_replacement inference

# %% [markdown]
# ## Contents
# 1. Imports
# 2. Runtime and configuration helpers
# 3. Inference status

# %% [markdown]
# ## 1. Imports

# %%
from __future__ import annotations

from IPython.display import display
from settings import ExperimentPaths, get_nested, load_config

# %% [markdown]
# ## 2. Runtime and configuration helpers

# %%
paths = ExperimentPaths()
config = load_config()

experiment_name = get_nested(config, "experiment.name")
route = get_nested(config, "experiment.route")
inference_config = get_nested(config, "inference") or {}

print(f"experiment={experiment_name}")
print(f"route={route}")
display(inference_config)

# %% [markdown]
# ## 3. Inference status

# %%
if inference_config.get("mode") != "not_applicable_train_feature_cache_only":
    raise ValueError("exp192 inference is intentionally disabled unless config is changed.")

print("No inference output is generated for this train feature cache experiment.")
print(
    "Downstream inference should regenerate raw test PF/Beam/likelihood-PF features "
    "with the same typewell_pct >= 0.50 hard-window generation code."
)
