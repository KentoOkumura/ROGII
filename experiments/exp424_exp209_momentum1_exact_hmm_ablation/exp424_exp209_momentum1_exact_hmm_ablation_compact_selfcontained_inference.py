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
# # exp424 exp209 momentum=1 exact-HMM ablation — inference guard
#
# exp424 is a train-side mechanism ablation. Stage 0 completed but failed its
# preregistered mechanism gates, so Stage 1 is ineligible and no inference or
# submission contract exists. This notebook remains a fail-closed placeholder.

# %% [markdown]
# ## Contents
#
# 1. Imports and notebook-safe configuration
# 2. Inference prohibition contract
# 3. Guarded execution

# %% [markdown]
# ## 1. Imports and notebook-safe configuration

# %%
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

EXPERIMENT_NAME = "exp424_exp209_momentum1_exact_hmm_ablation"
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
    candidates = (
        PACKAGE_DIR / "config.yaml",
        root / "experiments" / EXPERIMENT_NAME / "config.yaml",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("exp424 config.yaml was not found")


def load_config(path: Path | None = None) -> dict[str, Any]:
    resolved = config_path() if path is None else path
    value = yaml.safe_load(resolved.read_text()) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{resolved} must contain a YAML mapping")
    return value


# %% [markdown]
# ## 2. Inference prohibition contract


# %%
def validate_inference_disabled(config: Mapping[str, Any]) -> dict[str, Any]:
    if get_nested(config, "experiment.name") != EXPERIMENT_NAME:
        raise ValueError("wrong exp424 config")
    if get_nested(config, "experiment.route") != "pf_beam":
        raise ValueError("exp424 route must remain pf_beam")
    if bool(get_nested(config, "design.inference_authorized", True)):
        raise ValueError("design.inference_authorized must remain false")
    if bool(get_nested(config, "design.submission_authorized", True)):
        raise ValueError("design.submission_authorized must remain false")
    if bool(get_nested(config, "inference.enabled", True)):
        raise ValueError("inference.enabled must remain false")
    if bool(get_nested(config, "inference.create_submission", True)):
        raise ValueError("inference.create_submission must remain false")
    if bool(get_nested(config, "execution.create_submission", True)):
        raise ValueError("execution.create_submission must remain false")
    return {
        "experiment": EXPERIMENT_NAME,
        "route": "pf_beam",
        "stage0_completed": bool(get_nested(config, "design.kaggle_stage_0_completed", False)),
        "stage0_all_gates_pass": bool(
            get_nested(config, "design.kaggle_stage_0_all_gates_pass", False)
        ),
        "stage1_authorized": bool(get_nested(config, "design.kaggle_stage_1_authorized", False)),
        "inference_authorized": False,
        "submission_authorized": False,
        "unlock_condition": get_nested(config, "inference.unlock_condition"),
    }


def run_inference(config: Mapping[str, Any]) -> None:
    contract = validate_inference_disabled(config)
    raise RuntimeError(
        "exp424 inference is disabled: Stage 0 failed its mechanism gates, "
        "Stage 1 is ineligible, and no inference or submission contract exists. "
        f"Contract={json.dumps(contract, sort_keys=True)}"
    )


# %% [markdown]
# ## 3. Guarded execution

# %%
if __name__ == "__main__":
    CONFIG = load_config()
    CONTRACT = validate_inference_disabled(CONFIG)
    print(json.dumps(CONTRACT, sort_keys=True), flush=True)
    run_inference(CONFIG)
