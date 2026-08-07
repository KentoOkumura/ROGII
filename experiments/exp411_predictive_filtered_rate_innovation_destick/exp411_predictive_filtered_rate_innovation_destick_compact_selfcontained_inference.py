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
# # exp411 predictive→filtered rate-innovation de-stick — inference
#
# Stage 0 is a train-side fixed32 mechanism preflight. It does not authorize
# current-test inference or submission, and Stage 1 is separately locked.

# %% [markdown]
# ## Contents
#
# 1. Imports and notebook-safe configuration
# 2. Disabled inference contract
# 3. Fail-closed execution

# %% [markdown]
# ## 1. Imports and notebook-safe configuration

# %%
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import yaml

EXPERIMENT_NAME = "exp411_predictive_filtered_rate_innovation_destick"
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


def config_path() -> Path:
    root = find_project_root()
    for candidate in (
        PACKAGE_DIR / "config.yaml",
        root / "experiments" / EXPERIMENT_NAME / "config.yaml",
    ):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("exp411 config.yaml was not found")


def load_config(path: Path | None = None) -> dict[str, Any]:
    resolved = config_path() if path is None else path
    value = yaml.safe_load(resolved.read_text()) or {}
    if not isinstance(value, dict):
        raise ValueError("config must contain a YAML mapping")
    return value


# %% [markdown]
# ## 2. Disabled inference contract

# %%
def validate_disabled_inference(config: Mapping[str, Any]) -> dict[str, bool]:
    contract = {
        "stage_1_execution_approved": bool(
            get_nested(config, "design.stage_1_execution_approved", True)
        ),
        "inference_enabled": bool(
            get_nested(config, "design.inference_enabled", True)
        ),
        "submission_enabled": bool(
            get_nested(config, "design.submission_enabled", True)
        ),
    }
    if any(contract.values()):
        raise ValueError("exp411 Stage 0 inference contract must remain disabled")
    if get_nested(config, "execution.run_stage") != "stage_0_fixed32":
        raise ValueError("exp411 inference cannot run outside the fixed32 Stage 0 contract")
    return contract


def run_inference() -> None:
    raise RuntimeError(
        "exp411 inference is disabled: complete Stage 0, obtain separate Stage 1 "
        "approval, and obtain a later inference approval before creating predictions"
    )


# %% [markdown]
# ## 3. Fail-closed execution

# %%
if __name__ == "__main__":
    CONFIG = load_config()
    CONTRACT = validate_disabled_inference(CONFIG)
    print(
        json.dumps(
            {
                "event": "exp411_inference_disabled",
                "experiment": EXPERIMENT_NAME,
                **CONTRACT,
                "create_submission": False,
            },
            sort_keys=True,
        )
    )
    run_inference()
