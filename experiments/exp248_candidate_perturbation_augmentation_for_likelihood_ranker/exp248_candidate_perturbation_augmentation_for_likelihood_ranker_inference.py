# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # exp248 candidate perturbation augmentation for likelihood ranker — inference
#
# Inference is intentionally disabled. Augmented candidates are teacher rows only.
# A raw-test port is permitted only after clean OOF, hidden-like, long-tail, worst-well,
# and current-test feature-parity guards pass in this same experiment.

# %% [markdown]
# ## Contents
#
# 1. Configuration
# 2. Train-side-only guard

# %% [markdown]
# ## 1. Configuration

# %%
from __future__ import annotations

from settings import EXPERIMENT_NAME, get_nested, load_config

config = load_config()
print("Experiment:", EXPERIMENT_NAME)
print("Route:", get_nested(config, "experiment.route"))
print("Inference mode:", get_nested(config, "inference.mode"))
print("Selected variant:", get_nested(config, "inference.selected_variant"))
print("Submission enabled:", get_nested(config, "inference.submission"))

# %% [markdown]
# ## 2. Train-side-only guard

# %%
assert get_nested(config, "inference.mode") == "disabled_train_side_augmentation_audit_only"
assert get_nested(config, "inference.selected_variant") is None
assert get_nested(config, "inference.submission") is False
raise RuntimeError(
    "exp248 inference is disabled: candidate perturbations are training-only teacher views."
)
