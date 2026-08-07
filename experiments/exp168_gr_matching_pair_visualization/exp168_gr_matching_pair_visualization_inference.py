# %% [markdown]
# # exp168_gr_matching_pair_visualization inference

# %% [markdown]
# This experiment is a train-side GR matching visualization diagnostic. It does not
# generate predictions or a submission. The Kaggle execution entry is
# `exp168_gr_matching_pair_visualization_train.ipynb`.

# %%
from pathlib import Path

EXPERIMENT_NAME = "exp168_gr_matching_pair_visualization"

print(f"{EXPERIMENT_NAME}: inference notebook is intentionally a no-op.")
print("Run the train notebook to create GR matching pair visualizations.")
print("Working directory:", Path.cwd())
