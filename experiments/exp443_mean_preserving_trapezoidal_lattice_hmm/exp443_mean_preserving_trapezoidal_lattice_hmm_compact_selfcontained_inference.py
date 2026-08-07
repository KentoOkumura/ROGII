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
# # exp443 mean-preserving trapezoidal lattice HMM — inference guard
#
# exp443 is implemented but has not run fixed32 Stage 0. Inference, hidden-test
# HMM regeneration, submission creation, and Stage 1 remain disabled.

# %% [markdown]
# ## Contents
#
# 1. Imports and notebook-safe config loading
# 2. Disabled inference contract
# 3. Guarded orchestration

# %% [markdown]
# ## 1. Imports and notebook-safe config loading

# %%
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

EXPERIMENT_NAME = "exp443_mean_preserving_trapezoidal_lattice_hmm"
PACKAGE_DIR = Path.cwd()


def get_nested(mapping: Mapping[str, Any], dotted_key: str, default: Any = None) -> Any:
    value: Any = mapping
    for part in dotted_key.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return default
        value = value[part]
    return value


def find_project_root(start: Path = PACKAGE_DIR) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "project.yml").is_file():
            return candidate
    return start


def load_config() -> dict[str, Any]:
    root = find_project_root()
    candidates = (
        root / "experiments" / EXPERIMENT_NAME / "config.yaml",
        PACKAGE_DIR / "config.yaml",
    )
    for path in candidates:
        if path.is_file():
            value = yaml.safe_load(path.read_text()) or {}
            if not isinstance(value, dict):
                raise ValueError(f"{path} must contain a YAML mapping")
            if get_nested(value, "experiment.name") == EXPERIMENT_NAME:
                return value
    raise FileNotFoundError("exp443 config.yaml was not found")


# %% [markdown]
# ## 2. Disabled inference contract

# %%
def validate_inference_disabled(config: Mapping[str, Any]) -> dict[str, Any]:
    if get_nested(config, "experiment.name") != EXPERIMENT_NAME:
        raise ValueError("wrong exp443 config")
    contract = {
        "implementation_approved": bool(
            get_nested(config, "execution.implementation_authorized", False)
        ),
        "stage0_run_approved": bool(
            get_nested(config, "execution.stage0_run_authorized", False)
        ),
        "stage1_approved": bool(
            get_nested(config, "execution.stage1_run_authorized", False)
        ),
        "inference_enabled": bool(
            get_nested(config, "execution.inference_authorized", False)
        ),
        "submission_enabled": bool(
            get_nested(config, "execution.submission_authorized", False)
        ),
        "create_submission": bool(
            get_nested(config, "execution.create_submission", False)
        ),
    }
    if not contract["implementation_approved"]:
        raise RuntimeError("exp443 Stage 0 implementation is not approved")
    forbidden = {
        key: value
        for key, value in contract.items()
        if key
        in {
            "stage1_approved",
            "inference_enabled",
            "submission_enabled",
            "create_submission",
        }
        and value
    }
    if forbidden:
        raise ValueError(f"exp443 inference contract was unlocked: {forbidden}")
    return contract


def run_inference(config: Mapping[str, Any]) -> None:
    validate_inference_disabled(config)
    raise RuntimeError(
        "exp443 inference is disabled until separately authorized Stage 0 "
        "and Stage 1 promotion gates pass."
    )


# %% [markdown]
# ## 3. Guarded orchestration

# %%
CONFIG = load_config()
INFERENCE_CONTRACT = validate_inference_disabled(CONFIG)
print(
    json.dumps(
        {
            "experiment": EXPERIMENT_NAME,
            "status": get_nested(CONFIG, "experiment.status"),
            "inference_contract": INFERENCE_CONTRACT,
            "message": "Inference and submission remain fail-closed.",
        },
        indent=2,
        sort_keys=True,
    )
)
