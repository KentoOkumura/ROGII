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
# # exp299 base-posterior self-GR boundary handoff — inference guard
#
# Train-side evidence does not exist yet. This candidate records the explicit
# inference boundary and fails closed without loading raw test data, generating
# predictions, or writing a submission.

# %% [markdown]
# ## Contents
#
# 1. Imports and configuration lookup
# 2. Inference authorization contract
# 3. Fail-closed execution guard

# %% [markdown]
# ## 1. Imports and configuration lookup

# %%
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

EXPERIMENT_NAME = "exp299_base_posterior_self_gr_boundary_handoff"


def in_notebook_runtime() -> bool:
    try:
        return get_ipython() is not None  # type: ignore[name-defined]
    except NameError:
        return False


def load_config() -> dict[str, Any]:
    candidates = [
        Path.cwd() / "experiments" / EXPERIMENT_NAME / "config.yaml",
        Path.cwd() / "config.yaml",
    ]
    for candidate in candidates:
        if candidate.exists():
            payload = yaml.safe_load(candidate.read_text()) or {}
            if not isinstance(payload, dict):
                raise TypeError(f"Expected mapping in {candidate}")
            return payload
    raise FileNotFoundError(f"config.yaml not found: {[str(path) for path in candidates]}")


def get_nested(config: Mapping[str, Any], dotted_key: str, default: Any = None) -> Any:
    value: Any = config
    for part in dotted_key.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return default
        value = value[part]
    return value


# %% [markdown]
# ## 2. Inference authorization contract

# %%
def validate_inference_disabled(config: Mapping[str, Any]) -> dict[str, Any]:
    contract = {
        "experiment": get_nested(config, "experiment.name"),
        "route": get_nested(config, "experiment.route"),
        "train_run_completed": bool(get_nested(config, "execution.run_train", False)),
        "run_inference": bool(get_nested(config, "execution.run_inference", False)),
        "write_submission": bool(get_nested(config, "execution.write_submission", False)),
        "selected_candidate": get_nested(config, "inference.selected_candidate"),
        "canonical_train_notebook_adopted": bool(
            get_nested(config, "execution.canonical_train_notebook_adopted", False)
        ),
        "kaggle_cpu_push_approved": bool(
            get_nested(config, "execution.kaggle_cpu_push_approved", False)
        ),
    }
    if contract["experiment"] != EXPERIMENT_NAME:
        raise ValueError(f"unexpected experiment: {contract['experiment']}")
    if contract["route"] != "ensemble":
        raise ValueError(f"unexpected route: {contract['route']}")
    if contract["run_inference"] or contract["write_submission"]:
        raise ValueError("exp299 inference cannot be enabled before a new approved design")
    if contract["selected_candidate"] is not None:
        raise ValueError("exp299 selected_candidate must remain null before promotion")
    return contract


def run_inference(_: Mapping[str, Any]) -> None:
    raise RuntimeError(
        "exp299 inference is intentionally fail-closed: first run the separately approved "
        "train audit, pass every preregistered gate including the exp209 promotion bound, "
        "then obtain a raw-test-safe inference design and explicit approval."
    )


# %% [markdown]
# ## 3. Fail-closed execution guard

# %%
if in_notebook_runtime():
    CONFIG = load_config()
    PREFLIGHT = validate_inference_disabled(CONFIG)
    print(json.dumps(PREFLIGHT, indent=2, sort_keys=True))
    run_inference(CONFIG)
