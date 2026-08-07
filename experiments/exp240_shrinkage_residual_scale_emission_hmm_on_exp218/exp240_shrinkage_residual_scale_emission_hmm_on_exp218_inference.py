# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # exp240 shrinkage residual-scale emission HMM on exp218 — inference
#
# この実験は train-side scalar control / finite shrinkage ablation 専用である。
# raw-test residual scale、inference、submission は設計対象外とする。

# %% [markdown]
# ## Contents
# 1. Configuration contract
# 2. Disabled inference guard

# %%
from settings import load_config

config = load_config()
print(
    {
        "experiment": config["experiment"]["name"],
        "route": config["experiment"]["route"],
        "selected_stage": config["shrinkage"]["selected_stage"],
        "inference_enabled": config["experiment"]["inference_enabled"],
    }
)

# %% [markdown]
# ## 2. Disabled inference guard

# %%
if bool(config["experiment"].get("inference_enabled", False)):
    raise ValueError("exp240 inference cannot be enabled before a separate raw-test parity decision")

print("Train-side audit only: no prediction or submission file is generated.")
