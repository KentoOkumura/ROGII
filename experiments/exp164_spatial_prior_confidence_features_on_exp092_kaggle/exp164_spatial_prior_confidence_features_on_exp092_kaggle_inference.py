# %% [markdown]
# # exp164_spatial_prior_confidence_features_on_exp092_kaggle inference

# %% [markdown]
# ## Status
#
# This experiment is train-side only until OOF, worst-well, hidden-like stress, and
# raw-test spatial prior feature parity are reviewed. It intentionally does not generate
# `submission.csv`.

# %%
from settings import EXPERIMENT_NAME, get_nested, load_config

config = load_config()

print("Experiment:", EXPERIMENT_NAME)
print("Inference mode:", get_nested(config, "inference.mode"))
print("Selected variant:", get_nested(config, "inference.selected_variant"))
print("Notes:", get_nested(config, "inference.notes"))

# %% [markdown]
# Raw-test/full-train spatial prior feature parity is required before any inference port
# or submission.
