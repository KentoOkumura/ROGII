# %% [markdown]
# # exp353 Type-Well group quality feature preflight inference
#
# exp353 is a train-side zero-booster preflight.  Inference, raw-test feature
# regeneration, prediction, and submission remain disabled.

# %%
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

EXPERIMENT_NAME = "exp353_typewell_group_quality_feature_preflight"
KAGGLE_WORKING_ROOT = Path("/kaggle/working")


def get_nested(config: Mapping[str, Any], dotted_key: str) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def load_config() -> dict[str, Any]:
    candidates = [
        Path.cwd() / "config.yaml",
        KAGGLE_WORKING_ROOT / "config.yaml",
        Path.cwd() / "experiments" / EXPERIMENT_NAME / "config.yaml",
    ]
    for path in candidates:
        if not path.exists():
            continue
        payload = yaml.safe_load(path.read_text()) or {}
        if get_nested(payload, "experiment.name") == EXPERIMENT_NAME:
            return payload
    raise FileNotFoundError("exp353 config was not found")


def validate_disabled_inference(config: Mapping[str, Any]) -> dict[str, Any]:
    checks = {
        "implementation_inference_enabled": bool(
            get_nested(config, "implementation.inference_enabled")
        ),
        "implementation_submission_enabled": bool(
            get_nested(config, "implementation.submission_enabled")
        ),
        "run_inference": bool(get_nested(config, "execution.run_inference")),
        "create_submission": bool(get_nested(config, "execution.create_submission")),
        "inference_enabled": bool(get_nested(config, "inference.enabled")),
        "inference_create_submission": bool(
            get_nested(config, "inference.create_submission")
        ),
    }
    enabled = [name for name, value in checks.items() if value]
    if enabled:
        raise ValueError(
            "exp353 inference and submission must remain disabled; enabled flags: "
            f"{enabled}"
        )
    if bool(get_nested(config, "execution.run_stage_1")):
        raise ValueError("exp353 Stage 1 is separately gated and must remain disabled")
    return {
        "experiment": EXPERIMENT_NAME,
        "inference_enabled": False,
        "create_submission": False,
        "stage_1_enabled": False,
        "reason": (
            "Stage 0 is an OOF error-association preflight. Raw-test feature "
            "regeneration requires a later design boundary."
        ),
    }


CONFIG = load_config()
CONTRACT = validate_disabled_inference(CONFIG)
print(json.dumps(CONTRACT, indent=2, sort_keys=True), flush=True)
