# %% [markdown]
# # exp282 long-tail prediction-zone self-GR loop-closure inference
#
# exp282 is a train-side diagnostic only. This notebook deliberately fails
# closed and cannot create test predictions or a submission.

# %% [markdown]
# ## Contents
# 1. Imports and fixed disabled contract
# 2. Configuration check
# 3. Fail-closed inference boundary

# %% [markdown]
# ## 1. Imports and fixed disabled contract

# %%
from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

EXPERIMENT_NAME = "exp282_longtail_prediction_zone_self_gr_loop_closure_readout"


def in_notebook_runtime() -> bool:
    try:
        return get_ipython() is not None  # type: ignore[name-defined]
    except NameError:
        return False


EXECUTE_NOTEBOOK = os.environ.get("EXP282_IMPORT_ONLY", "0") != "1" and in_notebook_runtime()


# %% [markdown]
# ## 2. Configuration check


# %%
def get_nested(config: Mapping[str, Any], dotted_key: str) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
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
    candidates = (
        Path.cwd() / "config.yaml",
        project_root() / "experiments" / EXPERIMENT_NAME / "config.yaml",
    )
    for path in candidates:
        if not path.exists():
            continue
        value = yaml.safe_load(path.read_text()) or {}
        if isinstance(value, dict) and get_nested(value, "experiment.name") == EXPERIMENT_NAME:
            return value
    raise FileNotFoundError("exp282 config.yaml was not found")


def assert_inference_disabled(config: Mapping[str, Any]) -> None:
    if bool(get_nested(config, "inference.enabled")):
        raise ValueError("exp282 inference.enabled must remain false")
    if bool(get_nested(config, "inference.create_submission")):
        raise ValueError("exp282 create_submission must remain false")
    if bool(get_nested(config, "execution.inference")) or bool(
        get_nested(config, "execution.submission")
    ):
        raise ValueError("exp282 execution inference/submission flags must remain false")


# %% [markdown]
# ## 3. Fail-closed inference boundary


# %%
def fail_closed() -> None:
    raise RuntimeError(
        "exp282 is a train-side zero-booster readout. Inference, corrected predictions, "
        "and submission creation are intentionally disabled."
    )


if EXECUTE_NOTEBOOK:
    CONFIG = load_config()
    assert_inference_disabled(CONFIG)
    fail_closed()
