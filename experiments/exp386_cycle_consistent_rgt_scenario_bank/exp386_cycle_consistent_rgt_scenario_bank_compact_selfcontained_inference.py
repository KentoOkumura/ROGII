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
# # exp386 cycle-consistent RGT scenario bank inference
#
# exp386 is a train-side scenario-bank audit, not a deployable prediction. Test
# regeneration, inference, and submission remain fail-closed until all train-side
# gates pass and the scenario/reference-template manifest is pinned.

# %% [markdown]
# ## Contents
#
# 1. Imports and configuration
# 2. Fail-closed inference contract

# %% [markdown]
# ## 1. Imports and configuration

# %%
from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

EXPERIMENT_NAME = "exp386_cycle_consistent_rgt_scenario_bank"
IMPORT_ONLY_ENV = "EXP386_IMPORT_ONLY"
PACKAGE_DIR = Path.cwd()


def find_config() -> Path:
    candidates = [PACKAGE_DIR / "config.yaml"]
    candidates.extend(
        parent / "experiments" / EXPERIMENT_NAME / "config.yaml"
        for parent in (PACKAGE_DIR, *PACKAGE_DIR.parents)
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("exp386 config.yaml was not found")


def load_config() -> dict[str, Any]:
    value = yaml.safe_load(find_config().read_text()) or {}
    if not isinstance(value, dict):
        raise TypeError("config.yaml must contain a mapping")
    return value


def get_nested(mapping: Mapping[str, Any], dotted_key: str, default: Any = None) -> Any:
    value: Any = mapping
    for part in dotted_key.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return default
        value = value[part]
    return value


# %% [markdown]
# ## 2. Fail-closed inference contract

# %%
def validate_disabled_inference(config: Mapping[str, Any]) -> dict[str, Any]:
    if get_nested(config, "experiment.name") != EXPERIMENT_NAME:
        raise ValueError("wrong experiment config")
    contract = {
        "inference_enabled": bool(
            get_nested(config, "execution.inference_enabled", False)
        ),
        "submission_enabled": bool(
            get_nested(config, "execution.submission_enabled", False)
        ),
        "canonical_notebook_adoption_authorized": bool(
            get_nested(
                config,
                "execution.canonical_notebook_adoption_authorized",
                False,
            )
        ),
        "kaggle_execution_authorized": bool(
            get_nested(config, "execution.kaggle_execution_authorized", False)
        ),
    }
    if contract["inference_enabled"] or contract["submission_enabled"]:
        raise ValueError(
            "exp386 inference/submission cannot be enabled by this implementation"
        )
    return contract


def run_inference() -> None:
    config = load_config()
    validate_disabled_inference(config)
    raise RuntimeError(
        "exp386 inference is fail-closed: Stage 0/1/2 PASS, pinned scenario-bank "
        "and reference-GR-template SHAs, canonical adoption, and a separate "
        "inference approval are required"
    )


# %%
CONFIG_PREVIEW = load_config()
INFERENCE_CONTRACT = validate_disabled_inference(CONFIG_PREVIEW)
print("Experiment:", EXPERIMENT_NAME)
print("Inference contract:", INFERENCE_CONTRACT)

if os.environ.get(IMPORT_ONLY_ENV, "0") != "1":
    run_inference()
