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
# # exp384 fault-aware piecewise stratigraphic vector field inference
#
# Inference remains deliberately disabled.  The implementation-only request covers
# fold-safe CV code; test regeneration and submission require exp383/exp384 Stage
# 0/1 PASS, pinned artifacts, and a separate approval.

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

EXPERIMENT_NAME = "exp384_fault_aware_piecewise_stratigraphic_vector_field"
IMPORT_ONLY_ENV = "EXP384_IMPORT_ONLY"
PACKAGE_DIR = Path.cwd()


def find_config() -> Path:
    candidates = [PACKAGE_DIR / "config.yaml"]
    candidates.extend(
        parent
        / "experiments"
        / EXPERIMENT_NAME
        / "config.yaml"
        for parent in (PACKAGE_DIR, *PACKAGE_DIR.parents)
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("exp384 config.yaml was not found")


def load_config() -> dict[str, Any]:
    value = yaml.safe_load(find_config().read_text())
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
        "inference_enabled": bool(get_nested(config, "execution.inference_enabled", False)),
        "submission_enabled": bool(
            get_nested(config, "execution.submission_enabled", False)
        ),
        "parent_stage_pass_required": bool(
            get_nested(config, "execution.parent_stage_pass_required", True)
        ),
    }
    if contract["inference_enabled"] or contract["submission_enabled"]:
        raise ValueError(
            "exp384 inference/submission cannot be enabled before separate approval"
        )
    if not contract["parent_stage_pass_required"]:
        raise ValueError("exp383/exp384 stage PASS requirement cannot be disabled")
    return contract


def run_inference() -> None:
    config = load_config()
    validate_disabled_inference(config)
    raise RuntimeError(
        "exp384 inference is fail-closed: exp383 and exp384 Stage 0/1 PASS, "
        "pinned test artifacts, and a separate inference approval are required"
    )


# %%
CONFIG_PREVIEW = load_config()
INFERENCE_CONTRACT = validate_disabled_inference(CONFIG_PREVIEW)
print("Experiment:", EXPERIMENT_NAME)
print("Inference contract:", INFERENCE_CONTRACT)

if os.environ.get(IMPORT_ONLY_ENV, "0") != "1":
    run_inference()
