# %% [markdown]
# # exp311 Type-Well group prefix/suffix GR calibration readout — inference policy
#
# exp311 is a train-side readout. It never generates test predictions or a
# submission, even when its promotion gate passes.

# %% [markdown]
# ## Contents
# 1. Imports and config loading
# 2. Disabled-inference contract
# 3. Setup and policy preview

# %%
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

EXPERIMENT_NAME = "exp311_typewell_group_prefix_suffix_gr_calibration_readout"


def get_nested(config: dict[str, Any], dotted_key: str) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def project_root() -> Path:
    start = Path.cwd()
    for candidate in (start, *start.parents):
        if (candidate / "project.yml").exists() and (candidate / "AGENTS.md").exists():
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
        if get_nested(value, "experiment.name") == EXPERIMENT_NAME:
            return value
    raise FileNotFoundError("exp311 config.yaml was not found")


# %% [markdown]
# ## 2. Disabled-inference contract


# %%
def validate_disabled_inference(config: dict[str, Any]) -> dict[str, Any]:
    contract = {
        "experiment": get_nested(config, "experiment.name"),
        "route": get_nested(config, "experiment.route"),
        "scope": get_nested(config, "implementation.scope"),
        "inference_enabled": bool(get_nested(config, "implementation.inference_enabled")),
        "submission_enabled": bool(get_nested(config, "implementation.submission_enabled")),
        "inference_config_enabled": bool(get_nested(config, "inference.enabled")),
        "create_submission": bool(get_nested(config, "inference.create_submission")),
        "reason": get_nested(config, "inference.reason"),
    }
    if contract["experiment"] != EXPERIMENT_NAME or contract["route"] != "pf_beam":
        raise ValueError("exp311 inference policy loaded the wrong experiment contract")
    forbidden = (
        contract["inference_enabled"],
        contract["submission_enabled"],
        contract["inference_config_enabled"],
        contract["create_submission"],
    )
    if any(forbidden):
        raise ValueError("exp311 inference and submission must remain disabled")
    return contract


# %% [markdown]
# ## 3. Setup and policy preview


# %%
CONFIG = load_config()
INFERENCE_POLICY = validate_disabled_inference(CONFIG)
print(json.dumps(INFERENCE_POLICY, indent=2, sort_keys=True))
