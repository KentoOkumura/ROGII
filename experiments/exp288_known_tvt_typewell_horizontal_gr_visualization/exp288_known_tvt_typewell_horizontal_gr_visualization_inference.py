# %% [markdown]
# # exp288_known_tvt_typewell_horizontal_gr_visualization inference

# %% [markdown]
# ## Contents
# 1. Scope
# 2. Disabled inference entry

# %% [markdown]
# ## 1. Scope
#
# This experiment is a train-side known-TVT GR visualization diagnostic. It does
# not generate test features, predictions, or a submission. Run the train
# notebook to create one PNG per train well plus the manifest and HTML index.

# %% [markdown]
# ## 2. Disabled inference entry

# %%
from pathlib import Path

EXPERIMENT_NAME = "exp288_known_tvt_typewell_horizontal_gr_visualization"

print(f"{EXPERIMENT_NAME}: inference is intentionally disabled.")
print("Run the train notebook to generate known-TVT GR comparison figures.")
print("Working directory:", Path.cwd())
