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
# # exp383 all-TVT stratigraphic vector drift field inference
#
# Inference and submission are intentionally outside the approved scope.  This
# notebook records and enforces that boundary instead of creating a sample-copy
# submission that could be mistaken for a scientific result.

# %% [markdown]
# ## Contents
#
# 1. Imports and configuration
# 2. Disabled inference contract
# 3. Fail-closed execution

# %% [markdown]
# ## 1. Imports and configuration

# %%
from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

EXPERIMENT_NAME = "exp383_all_tvt_stratigraphic_vector_drift_field"
IMPORT_ONLY_ENV = "EXP383_IMPORT_ONLY"
PACKAGE_DIR = Path.cwd()


def get_nested(mapping: Mapping[str, Any], dotted_key: str, default: Any = None) -> Any:
    value: Any = mapping
    for part in dotted_key.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return default
        value = value[part]
    return value


def find_config() -> Path:
    local = PACKAGE_DIR / "config.yaml"
    if local.exists():
        return local
    for candidate in (PACKAGE_DIR, *PACKAGE_DIR.parents):
        path = candidate / "experiments" / EXPERIMENT_NAME / "config.yaml"
        if path.exists():
            return path
    raise FileNotFoundError("exp383 config.yaml was not found")


def load_config() -> dict[str, Any]:
    value = yaml.safe_load(find_config().read_text()) or {}
    if not isinstance(value, dict):
        raise ValueError("config.yaml must contain a mapping")
    return value


# %% [markdown]
# ## 2. Disabled inference contract

# %%
def validate_disabled_inference(config: Mapping[str, Any]) -> dict[str, Any]:
    if get_nested(config, "experiment.name") != EXPERIMENT_NAME:
        raise ValueError("wrong experiment config")
    contract = {
        "experiment": EXPERIMENT_NAME,
        "implementation_authorized": bool(
            get_nested(config, "execution.implementation_authorized", False)
        ),
        "inference_enabled": bool(
            get_nested(config, "execution.inference_enabled", False)
        ),
        "submission_enabled": bool(
            get_nested(config, "execution.submission_enabled", False)
        ),
        "required_before_inference": [
            "exp383 Stage 0 PASS",
            "exp383 Stage 1 PASS",
            "separate inference approval",
            "full-train artifact and SHA contract",
        ],
    }
    if not contract["implementation_authorized"]:
        raise ValueError("implementation authorization is not recorded")
    if contract["inference_enabled"] or contract["submission_enabled"]:
        raise ValueError("exp383 inference/submission must remain disabled")
    return contract


# %% [markdown]
# ## 3. Fail-closed execution

# %%
def run_inference() -> None:
    raise RuntimeError(
        "exp383 inference is fail-closed: train Stage 0/1 PASS and separate "
        "inference/submission approval are required"
    )


CONFIG = load_config()
CONTRACT = validate_disabled_inference(CONFIG)
print(CONTRACT)
if os.environ.get(IMPORT_ONLY_ENV) != "1":
    run_inference()
