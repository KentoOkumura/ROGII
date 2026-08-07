# %% [markdown]
# # exp290 piecewise datum physical smoother — inference disabled
#
# Stage 0 is a known-prefix pseudo-tail identifiability audit.  It does not
# authorize a full-suffix model, raw-test prediction, or submission.

# %% [markdown]
# ## Contents
# 1. Imports and configuration
# 2. Disabled inference contract
# 3. Fail-closed execution

# %% [markdown]
# ## 1. Imports and configuration

# %%
from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

EXPERIMENT_NAME = "exp290_piecewise_datum_physical_smoother"


def in_notebook_runtime() -> bool:
    try:
        return get_ipython() is not None  # type: ignore[name-defined]
    except NameError:
        return False


EXECUTE_NOTEBOOK = os.environ.get("EXP290_IMPORT_ONLY", "0") != "1" and in_notebook_runtime()


def get_nested(config: Mapping[str, Any], dotted_key: str, default: Any = None) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def load_config() -> dict[str, Any]:
    start = Path.cwd()
    candidates = [start / "config.yaml"]
    for parent in (start, *start.parents):
        candidates.append(parent / "experiments" / EXPERIMENT_NAME / "config.yaml")
    for path in candidates:
        if not path.exists():
            continue
        value = yaml.safe_load(path.read_text()) or {}
        if get_nested(value, "experiment.name") == EXPERIMENT_NAME:
            return value
    raise FileNotFoundError("exp290 config.yaml not found")


# %% [markdown]
# ## 2. Disabled inference contract

# %%
def validate_disabled_inference(config: Mapping[str, Any]) -> dict[str, Any]:
    contract = {
        "route": get_nested(config, "experiment.route"),
        "stage0_status": get_nested(config, "stages.stage0.implementation_status"),
        "stage1_status": get_nested(config, "stages.stage1.implementation_status"),
        "inference_enabled": bool(get_nested(config, "inference.enabled")),
        "create_submission": bool(get_nested(config, "inference.create_submission")),
        "kaggle_push_approved": bool(get_nested(config, "execution.kaggle_push_approved")),
    }
    if contract["route"] != "pf_beam":
        raise ValueError("exp290 route must remain pf_beam")
    if contract["inference_enabled"] or contract["create_submission"]:
        raise ValueError("exp290 inference must remain disabled before Stage 1 approval")
    return contract


def fail_closed() -> None:
    raise RuntimeError(
        "exp290 currently implements only the Stage 0 known-prefix pseudo-tail "
        "identifiability audit. No full-suffix TVT prediction or submission may be "
        "created before the Stage 0 guard passes and Stage 1 receives separate user "
        "approval."
    )


# %% [markdown]
# ## 3. Fail-closed execution

# %%
config = load_config()
contract = validate_disabled_inference(config)
if EXECUTE_NOTEBOOK:
    print("Experiment:", EXPERIMENT_NAME)
    print("Inference contract:", contract)
    fail_closed()
