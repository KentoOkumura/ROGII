# ---
# jupyter:
#   jupytext:
#     formats: py:percent
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
# # exp391 prefix-anchored mode-persistence HMM readout — inference
#
# exp391 is a train-side causal readout.  It does not define current-test mode
# identity, model artifacts, a submission candidate, or a permitted fallback
# contract.  This notebook records and enforces that boundary.

# %% [markdown]
# ## Contents
#
# 1. Imports and configuration
# 2. Train-side gate contract
# 3. Fail-closed inference boundary

# %% [markdown]
# ## 1. Imports and configuration

# %%
from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

EXPERIMENT_NAME = "exp391_prefix_anchored_mode_persistence_hmm_readout"
PACKAGE_DIR = Path.cwd()
EXECUTE_NOTEBOOK = os.environ.get("EXP391_IMPORT_ONLY", "0") != "1"


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


def find_config() -> Path:
    for start in (PACKAGE_DIR, *PACKAGE_DIR.parents):
        candidates = (
            start / "config.yaml",
            start / "experiments" / EXPERIMENT_NAME / "config.yaml",
        )
        for candidate in candidates:
            if candidate.exists():
                value = yaml.safe_load(candidate.read_text())
                if get_nested(value, "experiment.name") == EXPERIMENT_NAME:
                    return candidate
    raise FileNotFoundError("exp391 config.yaml was not found")


def load_config() -> dict[str, Any]:
    value = yaml.safe_load(find_config().read_text())
    if not isinstance(value, dict):
        raise TypeError("config.yaml must contain a mapping")
    return value


# %% [markdown]
# ## 2. Train-side gate contract

# %%
def inference_blockers(config: Mapping[str, Any]) -> list[str]:
    blockers = [
        "exp391 is a train-side posterior and downstream-path readout",
        "current-test stable-mode lineage has not been validated",
        "no saved current-test model or decoder manifest is approved",
        "no submission or row-fallback contract is in scope",
    ]
    if not bool(get_nested(config, "stage_b.enabled", False)):
        blockers.append("Stage B full train-side readout is disabled")
    if not bool(get_nested(config, "execution.stage_b_run_approved", False)):
        blockers.append("Stage B full train-side run is not approved")
    if not bool(get_nested(config, "execution.inference_approved", False)):
        blockers.append("inference is not approved")
    if not bool(get_nested(config, "execution.submission_approved", False)):
        blockers.append("submission is not approved")
    return blockers


def validate_inference_disabled(config: Mapping[str, Any]) -> None:
    if get_nested(config, "experiment.name") != EXPERIMENT_NAME:
        raise ValueError("experiment name differs from exp391")
    if get_nested(config, "experiment.route") != "pf_beam":
        raise ValueError("exp391 route must remain pf_beam")
    if bool(get_nested(config, "experiment.inference_enabled", False)):
        raise ValueError("experiment.inference_enabled must remain false")
    if bool(get_nested(config, "execution.run_inference", False)):
        raise ValueError("execution.run_inference must remain false")
    if bool(get_nested(config, "execution.create_submission", False)):
        raise ValueError("execution.create_submission must remain false")
    if bool(get_nested(config, "execution.submit_to_kaggle", False)):
        raise ValueError("execution.submit_to_kaggle must remain false")


# %% [markdown]
# ## 3. Fail-closed inference boundary

# %%
def main() -> None:
    config = load_config()
    validate_inference_disabled(config)
    blockers = inference_blockers(config)
    print(
        json.dumps(
            {
                "experiment": EXPERIMENT_NAME,
                "status": "inference_disabled_fail_closed",
                "blockers": blockers,
            },
            indent=2,
            sort_keys=True,
        )
    )
    raise RuntimeError(
        "exp391 inference/submission is outside the approved experiment scope"
    )


if EXECUTE_NOTEBOOK:
    main()
