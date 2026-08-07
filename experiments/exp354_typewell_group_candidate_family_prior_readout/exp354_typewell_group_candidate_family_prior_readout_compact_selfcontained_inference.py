# %% [markdown]
# # exp354 Type-Well group candidate-family prior readout inference
#
# Stage 0 is a train-side diagnostic only.  This fail-closed notebook documents
# the inference boundary and refuses raw-test prior generation, selector use,
# prediction, or submission creation.

# %% [markdown]
# ## Contents
# 1. Imports
# 2. Notebook-safe config helpers
# 3. Disabled inference contract
# 4. Setup and explicit stop

# %%
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import yaml


EXPERIMENT_NAME = "exp354_typewell_group_candidate_family_prior_readout"
PACKAGE_DIR = Path.cwd()


# %%
def get_nested(config: Mapping[str, Any], dotted_key: str, default: Any = None) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def resolve_config_path() -> Path:
    candidates = [
        PACKAGE_DIR / "config.yaml",
        PACKAGE_DIR / "experiments" / EXPERIMENT_NAME / "config.yaml",
        Path("/kaggle/working/config.yaml"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("exp354 config.yaml was not found")


def read_config() -> dict[str, Any]:
    value = yaml.safe_load(resolve_config_path().read_text()) or {}
    if not isinstance(value, dict):
        raise ValueError("exp354 config must be a YAML mapping")
    return value


# %% [markdown]
# ## 3. Disabled inference contract

# %%
def validate_disabled_inference(config: Mapping[str, Any]) -> dict[str, Any]:
    expected_false = {
        "implementation.stage_1_implemented": False,
        "implementation.inference_enabled": False,
        "implementation.submission_enabled": False,
        "execution.run_stage_1": False,
        "execution.run_inference": False,
        "execution.create_submission": False,
        "inference.enabled": False,
        "inference.create_submission": False,
    }
    for key, expected in expected_false.items():
        if get_nested(config, key) is not expected:
            raise ValueError(f"exp354 {key} must remain disabled")
    if get_nested(config, "execution_contract.stage_0.boosters") != 0:
        raise ValueError("exp354 Stage 0 must remain a zero-booster readout")
    if get_nested(config, "execution_contract.stage_1_if_pass.selector_models") != 40:
        raise ValueError("exp354 conditional Stage 1 model count changed")
    if get_nested(config, "execution_contract.parent_control_retraining") is not False:
        raise ValueError("exp354 parent/control retraining must remain disabled")
    return {
        "experiment": EXPERIMENT_NAME,
        "stage_0_only": True,
        "stage_1_implemented": False,
        "inference_enabled": False,
        "create_submission": False,
        "stage_0_boosters": 0,
        "conditional_stage_1_models": 40,
        "control_models": 0,
    }


def stop_disabled_inference(config: Mapping[str, Any]) -> None:
    validate_disabled_inference(config)
    raise RuntimeError(
        "exp354 Stage 1, inference, and submission are disabled. "
        "A Stage 0 PASS would still require separate user approval."
    )


# %% [markdown]
# ## 4. Setup and explicit stop

# %%
CONFIG = read_config()
INFERENCE_CONTRACT = validate_disabled_inference(CONFIG)
print("Experiment:", get_nested(CONFIG, "experiment.name"))
print("Route:", get_nested(CONFIG, "experiment.route"))
print("Status:", get_nested(CONFIG, "experiment.status"))
print("Inference contract:", INFERENCE_CONTRACT)

if os.environ.get("EXP354_IMPORT_ONLY") != "1":
    stop_disabled_inference(CONFIG)
