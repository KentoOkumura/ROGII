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
# # exp406 loop-closed multi-well RGT fixed16 Stage 0 inference
#
# exp406 Stage 0 is a fixed16 train-side graph diagnostic. It intentionally
# persists no unknown-suffix prediction and cannot create a submission.

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

EXPERIMENT_NAME = "exp406_loop_closed_multiwell_rgt_fixed16_stage0"
IMPORT_ONLY_ENV = "EXP406_IMPORT_ONLY"
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
    raise FileNotFoundError("exp406 config.yaml was not found")


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
def validate_disabled_inference(config: Mapping[str, Any]) -> dict[str, bool]:
    if get_nested(config, "experiment.name") != EXPERIMENT_NAME:
        raise ValueError("wrong experiment config")
    contract = {
        "experiment_inference_enabled": bool(
            get_nested(config, "experiment.inference_enabled", False)
        ),
        "experiment_submission_enabled": bool(
            get_nested(config, "experiment.submission_enabled", False)
        ),
        "execution_inference_authorized": bool(
            get_nested(config, "execution.inference_authorized", False)
        ),
        "execution_submission_authorized": bool(
            get_nested(config, "execution.submission_authorized", False)
        ),
        "full_oof_stage1_authorized": bool(
            get_nested(config, "execution.full_oof_stage1_authorized", False)
        ),
    }
    if any(contract.values()):
        raise ValueError(
            "exp406 Stage 0 inference, submission, and full OOF must remain disabled"
        )
    return contract


def run_inference() -> None:
    config = load_config()
    validate_disabled_inference(config)
    raise RuntimeError(
        "exp406 inference is fail-closed: fixed16 Stage 0 produces diagnostics "
        "only. Full-OOF Stage 1 requires a PASS, a new design, and separate "
        "approval; current-test inference and submission are not implemented."
    )


# %%
CONFIG_PREVIEW = load_config()
INFERENCE_CONTRACT = validate_disabled_inference(CONFIG_PREVIEW)
print("Experiment:", EXPERIMENT_NAME)
print("Inference contract:", INFERENCE_CONTRACT)

if os.environ.get(IMPORT_ONLY_ENV, "0") != "1":
    run_inference()
