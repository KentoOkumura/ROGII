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
# # exp445 TVT-to-U coordinate parity exact HMM — inference guard
#
# exp445 is a truth-free technical coordinate-parity audit. It never produces
# a hidden-test prediction or submission. This notebook candidate records and
# enforces that boundary.

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
from pathlib import Path
from typing import Any, Mapping

import yaml

EXPERIMENT_NAME = "exp445_tvt_to_u_coordinate_parity_exact_hmm"
PACKAGE_DIR = Path.cwd()


def get_nested(
    mapping: Mapping[str, Any],
    dotted_key: str,
    default: Any = None,
) -> Any:
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
    raise FileNotFoundError("exp445 config.yaml was not found")


# %% [markdown]
# ## 2. Disabled inference contract

# %%
def validate_inference_disabled(
    config: Mapping[str, Any],
) -> dict[str, bool]:
    if get_nested(config, "experiment.name") != EXPERIMENT_NAME:
        raise ValueError("wrong exp445 config")
    contract = {
        "implementation_authorized": bool(
            get_nested(config, "design.implementation_authorized", False)
        ),
        "canonical_notebook_adoption_authorized": bool(
            get_nested(
                config,
                "design.canonical_notebook_adoption_authorized",
                True,
            )
        ),
        "kaggle_package_authorized": bool(
            get_nested(config, "design.kaggle_package_authorized", True)
        ),
        "kaggle_run_authorized": bool(
            get_nested(config, "design.kaggle_run_authorized", True)
        ),
        "inference_authorized": bool(
            get_nested(config, "design.inference_authorized", True)
        ),
        "submission_authorized": bool(
            get_nested(config, "design.submission_authorized", True)
        ),
        "create_submission": bool(
            get_nested(config, "execution.create_submission", True)
        ),
    }
    if not contract["implementation_authorized"]:
        raise RuntimeError("exp445 implementation is not authorized")
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
        raise ValueError(f"exp445 inference contract was unlocked: {forbidden}")
    return contract


def run_inference(config: Mapping[str, Any]) -> None:
    validate_inference_disabled(config)
    raise RuntimeError(
        "exp445 has no inference stage: it is a fixed32 truth-free coordinate "
        "parity audit and cannot create a hidden-test prediction or submission."
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
                "event": "exp445_inference_disabled",
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
