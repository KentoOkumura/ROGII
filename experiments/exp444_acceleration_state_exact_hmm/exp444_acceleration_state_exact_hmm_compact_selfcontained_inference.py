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
# # exp444 acceleration-state exact HMM — inference guard
#
# exp444 is implemented only through the fixed4 train-side Stage 0A candidate.
# Hidden-test regeneration, submission creation, and inference remain disabled.

# %% [markdown]
# ## Contents
#
# 1. Imports and notebook-safe configuration
# 2. Fail-closed inference contract
# 3. Guarded orchestration

# %% [markdown]
# ## 1. Imports and notebook-safe configuration

# %%
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

EXPERIMENT_NAME = "exp444_acceleration_state_exact_hmm"
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
    raise FileNotFoundError("exp444 config.yaml was not found")


def load_config(path: Path | None = None) -> dict[str, Any]:
    resolved = config_path() if path is None else path
    value = yaml.safe_load(resolved.read_text()) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{resolved} must contain a YAML mapping")
    return value


# %% [markdown]
# ## 2. Fail-closed inference contract

# %%
def validate_inference_disabled(config: Mapping[str, Any]) -> dict[str, bool]:
    if get_nested(config, "experiment.name") != EXPERIMENT_NAME:
        raise ValueError("wrong exp444 config")
    contract = {
        "implementation_authorized": bool(
            get_nested(config, "execution.implementation_authorized", False)
        ),
        "canonical_notebook_adoption_authorized": bool(
            get_nested(
                config,
                "execution.canonical_notebook_adoption_authorized",
                False,
            )
        ),
        "kaggle_package_authorized": bool(
            get_nested(config, "execution.kaggle_package_authorized", False)
        ),
        "stage0a_run_authorized": bool(
            get_nested(config, "execution.stage0a_run_authorized", False)
        ),
        "stage0b_run_authorized": bool(
            get_nested(config, "execution.stage0b_run_authorized", True)
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
        raise RuntimeError("exp444 implementation is not authorized")
    forbidden = {
        key: contract[key]
        for key in (
            "inference_authorized",
            "submission_authorized",
            "create_submission",
        )
        if contract[key]
    }
    if forbidden:
        raise ValueError(f"exp444 inference contract was unlocked: {forbidden}")
    if get_nested(config, "inference.mode") != "disabled_implementation_only":
        raise ValueError("exp444 inference mode changed")
    return contract


def run_inference(config: Mapping[str, Any]) -> None:
    validate_inference_disabled(config)
    raise RuntimeError(
        "exp444 inference is disabled until all train-side gates pass and a "
        "later, separate inference authorization is recorded."
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
                "event": "exp444_inference_disabled",
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
