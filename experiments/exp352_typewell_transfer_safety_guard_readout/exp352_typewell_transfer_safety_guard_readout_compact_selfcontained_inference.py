# %% [markdown]
# # exp352 Type-Well transfer safety guard inference
#
# exp352 is a train-side zero-model diagnostic. Inference and submission remain
# explicitly fail-closed even if Stage 0 later passes.

# %% [markdown]
# ## Contents
# 1. Imports
# 2. Runtime and configuration helpers
# 3. Disabled inference contract

# %%
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

EXPERIMENT_NAME = "exp352_typewell_transfer_safety_guard_readout"
KAGGLE_WORKING_ROOT = Path("/kaggle/working")


# %% [markdown]
# ## 2. Runtime and configuration helpers

# %%
def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text()) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return payload


def get_nested(config: Mapping[str, Any], dotted_key: str) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def project_root() -> Path:
    for start in (Path.cwd(), KAGGLE_WORKING_ROOT):
        for candidate in (start, *start.parents):
            if (candidate / "project.yml").exists():
                return candidate
    return Path.cwd()


def load_experiment_config() -> dict[str, Any]:
    root = project_root()
    candidates = [
        Path.cwd() / "config.yaml",
        KAGGLE_WORKING_ROOT / "config.yaml",
        root / "experiments" / EXPERIMENT_NAME / "config.yaml",
    ]
    for path in candidates:
        config = read_yaml(path)
        if get_nested(config, "experiment.name") == EXPERIMENT_NAME:
            return config
    raise FileNotFoundError(f"exp352 config not found in {[str(path) for path in candidates]}")


# %% [markdown]
# ## 3. Disabled inference contract

# %%
def validate_disabled_inference(config: Mapping[str, Any]) -> dict[str, Any]:
    contract = {
        "experiment": get_nested(config, "experiment.name"),
        "route": get_nested(config, "experiment.route"),
        "inference_enabled": bool(get_nested(config, "inference.enabled")),
        "implementation_inference_enabled": bool(
            get_nested(config, "implementation.inference_enabled")
        ),
        "run_inference": bool(get_nested(config, "execution.run_inference")),
        "create_submission": bool(get_nested(config, "execution.create_submission")),
        "submission_enabled": bool(get_nested(config, "implementation.submission_enabled")),
    }
    if contract["experiment"] != EXPERIMENT_NAME or contract["route"] != "pf_beam":
        raise ValueError("wrong experiment or route for exp352 inference")
    enabled = [
        key
        for key, value in contract.items()
        if key
        in {
            "inference_enabled",
            "implementation_inference_enabled",
            "run_inference",
            "create_submission",
            "submission_enabled",
        }
        and bool(value)
    ]
    if enabled:
        raise ValueError(f"exp352 inference/submission must remain disabled: {enabled}")
    return contract


CONFIG = load_experiment_config()
DISABLED_INFERENCE_CONTRACT = validate_disabled_inference(CONFIG)
print(json.dumps(DISABLED_INFERENCE_CONTRACT, indent=2, sort_keys=True), flush=True)
