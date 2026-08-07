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
# # exp442 symmetric broad-jump rate-transition exact HMM — inference guard
#
# exp442 currently authorizes canonical train notebook adoption, Kaggle private
# CPU packaging, and fixed32 Stage 0. Stage 1, hidden-test regeneration,
# inference, and submission remain separately locked.

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

EXPERIMENT_NAME = "exp442_symmetric_broad_jump_rate_transition_hmm"
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
    for path in (
        PACKAGE_DIR / "config.yaml",
        root / "experiments" / EXPERIMENT_NAME / "config.yaml",
    ):
        if path.is_file():
            value = yaml.safe_load(path.read_text()) or {}
            if not isinstance(value, dict):
                raise ValueError(f"{path} must contain a YAML mapping")
            if get_nested(value, "experiment.name") == EXPERIMENT_NAME:
                return value
    raise FileNotFoundError("exp442 config.yaml was not found")


# %% [markdown]
# ## 2. Disabled inference contract

# %%
def validate_inference_disabled(config: Mapping[str, Any]) -> dict[str, bool]:
    if get_nested(config, "experiment.name") != EXPERIMENT_NAME:
        raise ValueError("wrong exp442 config")
    contract = {
        "implementation_authorized": bool(
            get_nested(config, "execution.implementation_authorized", False)
        ),
        "canonical_notebook_adoption_authorized": bool(
            get_nested(
                config,
                "execution.canonical_notebook_adoption_authorized",
                True,
            )
        ),
        "kaggle_package_authorized": bool(
            get_nested(config, "execution.kaggle_package_authorized", True)
        ),
        "stage0_run_authorized": bool(
            get_nested(config, "execution.stage0_run_authorized", True)
        ),
        "stage1_run_authorized": bool(
            get_nested(config, "execution.stage1_run_authorized", True)
        ),
        "inference_authorized": bool(
            get_nested(config, "execution.inference_authorized", True)
        ),
        "submission_authorized": bool(
            get_nested(config, "execution.submission_authorized", True)
        ),
        "create_submission": bool(
            get_nested(config, "execution.create_submission", True)
        ),
    }
    if not contract["implementation_authorized"]:
        raise RuntimeError("exp442 implementation is not authorized")
    required_train_stage0 = {
        key: value
        for key, value in contract.items()
        if key
        in {
            "canonical_notebook_adoption_authorized",
            "kaggle_package_authorized",
            "stage0_run_authorized",
        }
        and not value
    }
    if required_train_stage0:
        raise ValueError(
            "exp442 authorized train Stage 0 contract is incomplete: "
            f"{required_train_stage0}"
        )
    forbidden = {
        key: value
        for key, value in contract.items()
        if key
        in {
            "stage1_run_authorized",
            "inference_authorized",
            "submission_authorized",
            "create_submission",
        }
        and value
    }
    if forbidden:
        raise ValueError(f"exp442 inference contract was unlocked: {forbidden}")
    return contract


def run_inference(config: Mapping[str, Any]) -> None:
    validate_inference_disabled(config)
    raise RuntimeError(
        "exp442 inference is disabled: finish the authorized fixed32 Stage 0, "
        "pass every gate, obtain separate Stage 1 approval, and then obtain a "
        "later inference approval."
    )


# %% [markdown]
# ## 3. Guarded orchestration

# %%
if __name__ == "__main__":
    CONFIG = load_config()
    CONTRACT = validate_inference_disabled(CONFIG)
    print(
        json.dumps(
            {
                "event": "exp442_inference_disabled",
                "experiment": EXPERIMENT_NAME,
                "status": get_nested(CONFIG, "experiment.status"),
                "inference_contract": CONTRACT,
                "message": "Inference and submission remain fail-closed.",
            },
            indent=2,
            sort_keys=True,
        )
    )
    run_inference(CONFIG)
