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
# # exp267 well-segment candidate divergence signature cluster — inference
#
# exp267はtrain-side Stage A 0-booster監査専用であり、current-test cluster、selector、
# `submission.csv`を生成しない。Stage A全guard通過とconditional Stage Bの別承認後も、
# inferenceは同じ実験内で設計・承認してから実装する。

# %% [markdown]
# ## Contents
#
# 1. Setup and paths
# 2. Disabled-stage guard

# %% [markdown]
# ## 1. Setup and paths

# %%
from settings import EXPERIMENT_NAME, ExperimentPaths, load_config

paths = ExperimentPaths()
config = load_config()
print(
    {
        "experiment": EXPERIMENT_NAME,
        "route": config["experiment"]["route"],
        "stage": config["execution"]["stage"],
        "inference_enabled": config["execution"]["inference_enabled"],
    }
)

# %% [markdown]
# ## 2. Disabled-stage guard

# %%
assert config["execution"]["inference_enabled"] is False
assert config["model"]["conditional_stage_b"]["enabled"] is False
raise RuntimeError(
    "exp267 inference is intentionally disabled: Stage A has not run and no submission is allowed."
)
