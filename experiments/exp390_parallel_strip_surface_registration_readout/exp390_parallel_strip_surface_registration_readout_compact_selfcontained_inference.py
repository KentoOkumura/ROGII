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
# # exp390 parallel strip surface registration readout — inference
#
# Inference is deliberately fail-closed. The experiment must pass both the
# scientific-support and promotion-safety gates, then receive separate approval.

# %% [markdown]
# ## Contents
#
# 1. Configuration check
# 2. Fail-closed inference guard

# %%
from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

EXPERIMENT_NAME = "exp390_parallel_strip_surface_registration_readout"
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


def find_config() -> Path:
    candidates = (
        PACKAGE_DIR / "config.yaml",
        PACKAGE_DIR / "experiments" / EXPERIMENT_NAME / "config.yaml",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    for parent in PACKAGE_DIR.parents:
        candidate = parent / "experiments" / EXPERIMENT_NAME / "config.yaml"
        if candidate.exists():
            return candidate
    raise FileNotFoundError("exp390 config.yaml was not found")


def load_config() -> dict[str, Any]:
    value = yaml.safe_load(find_config().read_text())
    if not isinstance(value, dict):
        raise TypeError("config.yaml must contain a mapping")
    return value


def assert_inference_authorized(config: Mapping[str, Any]) -> None:
    if not bool(get_nested(config, "execution.inference_enabled", False)):
        raise RuntimeError(
            "exp390 inference is disabled: Stage 2 scientific-support and "
            "promotion-safety gates plus separate user approval are required"
        )
    if not bool(get_nested(config, "execution.submission_enabled", False)):
        raise RuntimeError("exp390 submission is disabled")


# %%
CONFIG = load_config()
print("Experiment:", EXPERIMENT_NAME)
print("Route:", get_nested(CONFIG, "experiment.route"))
print("Inference enabled:", get_nested(CONFIG, "execution.inference_enabled"))
print("Submission enabled:", get_nested(CONFIG, "execution.submission_enabled"))

if os.environ.get("EXP390_IMPORT_ONLY", "0") != "1":
    assert_inference_authorized(CONFIG)
    raise NotImplementedError(
        "Current-test strip generation is intentionally outside the authorized "
        "exp390 implementation scope."
    )
