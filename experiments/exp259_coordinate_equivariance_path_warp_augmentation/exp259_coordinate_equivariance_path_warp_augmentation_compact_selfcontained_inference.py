# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # exp259 coordinate equivariance / path warp augmentation — inference
#
# Inference is intentionally disabled. The optional exact-datum train stage must first
# beat the separately trained exp251 corrected 295-feature control on all safety guards.
# Approximate path warps, direct PF/HMM replacement, and submission remain disabled.

# %% [markdown]
# ## Contents
# 1. Experiment contract
# 2. Explicit inference guard

# %% [markdown]
# ## 1. Experiment contract

# %%
from pathlib import Path

import yaml

EXPERIMENT_NAME = "exp259_coordinate_equivariance_path_warp_augmentation"


def find_config() -> Path:
    cwd = Path.cwd()
    for candidate in [cwd, *cwd.parents]:
        direct = candidate / "config.yaml"
        if direct.exists() and EXPERIMENT_NAME in direct.read_text():
            return direct
        nested = candidate / "experiments" / EXPERIMENT_NAME / "config.yaml"
        if nested.exists():
            return nested
    raise FileNotFoundError(f"could not locate {EXPERIMENT_NAME}/config.yaml")


config = yaml.safe_load(find_config().read_text()) or {}
assert config["experiment"]["name"] == EXPERIMENT_NAME
assert config["experiment"]["route"] == "ensemble"
assert config["model"]["trains_new_boosters"] is True
assert config["model"]["planned_boosters"] == 10
assert config["model"]["control_retraining"] is False
assert config["execution"]["inference_enabled"] is False
assert config["submission"]["enabled"] is False

print(
    {
        "experiment": EXPERIMENT_NAME,
        "stage": config["execution"]["stage"],
        "inference_enabled": config["execution"]["inference_enabled"],
        "submission_enabled": config["submission"]["enabled"],
    }
)


# %% [markdown]
# ## 2. Explicit inference guard

# %%
raise RuntimeError(
    "exp259 inference is disabled until exact-datum training beats the fixed exp251 "
    "295-feature clean control on every configured guard."
)
