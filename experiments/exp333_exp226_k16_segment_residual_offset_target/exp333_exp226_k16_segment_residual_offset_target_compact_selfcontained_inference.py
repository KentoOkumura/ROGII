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
# # exp333 exp226 K16 segment residual offset target — inference
#
# Inference is intentionally unavailable. Stage 1 train execution authorization
# is independent of inference authorization. A current-test path may be
# implemented only after Stage 1 passes every scientific gate, the exp263
# comparison threshold is met, and the user separately approves inference.

# %% [markdown]
# ## Contents
# 1. Imports and path helpers
# 2. Configuration lookup
# 3. Fail-closed inference contract
# 4. Disabled inference report

# %% [markdown]
# ## 1. Imports and path helpers

# %%
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

EXPERIMENT_NAME = "exp333_exp226_k16_segment_residual_offset_target"


def project_root() -> Path:
    start = Path.cwd()
    for candidate in (start, *start.parents):
        if (candidate / "project.yml").exists() and (candidate / "AGENTS.md").exists():
            return candidate
    return start


# %% [markdown]
# ## 2. Configuration lookup

# %%
def load_config() -> dict[str, Any]:
    candidates = (
        Path.cwd() / "config.yaml",
        project_root() / "experiments" / EXPERIMENT_NAME / "config.yaml",
    )
    for path in candidates:
        if path.exists():
            value = yaml.safe_load(path.read_text()) or {}
            if not isinstance(value, dict):
                raise ValueError(f"{path} must contain a YAML mapping")
            return value
    raise FileNotFoundError("exp333 config.yaml was not found")


# %% [markdown]
# ## 3. Fail-closed inference contract

# %%
def validate_disabled_inference(config: dict[str, Any]) -> dict[str, Any]:
    implementation = config.get("implementation", {})
    execution = config.get("execution_contract", {})
    inference = config.get("inference", {})
    if implementation.get("stage_1_enabled") is not True:
        raise ValueError("Stage 1 train implementation must remain present")
    if implementation.get("inference_enabled") is not False:
        raise ValueError("implementation.inference_enabled must remain false")
    if implementation.get("submission_enabled") is not False:
        raise ValueError("implementation.submission_enabled must remain false")
    if inference.get("enabled") is not False or inference.get("create_submission") is not False:
        raise ValueError("exp333 inference and submission must remain disabled")
    if execution.get("inference_approved") is not False:
        raise ValueError("exp333 inference has not been separately approved")
    if execution.get("submission_approved") is not False:
        raise ValueError("exp333 submission has not been separately approved")
    return {
        "experiment": EXPERIMENT_NAME,
        "status": "disabled_fail_closed",
        "stage_0_is_diagnostic_only": True,
        "stage_1_train_implemented": True,
        "stage_1_train_execution_authorized": bool(
            execution.get("stage_1_run_approved")
        ),
        "deployable_model_available": False,
        "submission_created": False,
        "reason": inference.get("reason"),
    }


# %% [markdown]
# ## 4. Disabled inference report

# %%
CONFIG = load_config()
REPORT = validate_disabled_inference(CONFIG)
print(json.dumps(REPORT, indent=2, sort_keys=True))
