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
# # exp312 conditional GR emission table — inference policy
#
# exp312 is a train-side candidate-rank audit. It never regenerates candidate
# paths, runs a decoder, predicts hidden TVT, or creates a submission.

# %% [markdown]
# ## Contents
# 1. Imports and configuration
# 2. Disabled-inference contract
# 3. Policy preview

# %%
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

EXPERIMENT_NAME = "exp312_typewell_group_conditional_gr_emission_table"


def get_nested(config: dict[str, Any], dotted_key: str) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def project_root() -> Path:
    start = Path.cwd()
    for candidate in (start, *start.parents):
        if (candidate / "project.yml").exists():
            return candidate
    return start


def load_config() -> dict[str, Any]:
    for path in (
        Path.cwd() / "config.yaml",
        project_root() / "experiments" / EXPERIMENT_NAME / "config.yaml",
    ):
        if not path.exists():
            continue
        value = yaml.safe_load(path.read_text()) or {}
        if get_nested(value, "experiment.name") == EXPERIMENT_NAME:
            return value
    raise FileNotFoundError("exp312 config.yaml was not found")


# %% [markdown]
# ## 2. Disabled-inference contract


# %%
def validate_disabled_inference(config: dict[str, Any]) -> dict[str, Any]:
    contract = {
        "experiment": get_nested(config, "experiment.name"),
        "route": get_nested(config, "experiment.route"),
        "scope": get_nested(config, "implementation.scope"),
        "inference_enabled": bool(get_nested(config, "implementation.inference_enabled")),
        "submission_enabled": bool(get_nested(config, "implementation.submission_enabled")),
        "inference_config_enabled": bool(get_nested(config, "inference.enabled")),
        "create_submission": bool(get_nested(config, "inference.create_submission")),
        "reason": get_nested(config, "inference.reason"),
    }
    if contract["experiment"] != EXPERIMENT_NAME or contract["route"] != "pf_beam":
        raise ValueError("exp312 inference policy loaded the wrong experiment contract")
    if any(
        (
            contract["inference_enabled"],
            contract["submission_enabled"],
            contract["inference_config_enabled"],
            contract["create_submission"],
        )
    ):
        raise ValueError("exp312 inference and submission must remain disabled")
    return contract


# %% [markdown]
# ## 3. Policy preview

# %%
CONFIG = load_config()
INFERENCE_POLICY = validate_disabled_inference(CONFIG)
print(json.dumps(INFERENCE_POLICY, indent=2, sort_keys=True))
