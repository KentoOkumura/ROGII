# %% [markdown]
# # exp401 exp368 weak-risk candidate-advantage inference
#
# Inference is intentionally unavailable. Stage 0 is a train-side diagnostic
# with zero TVT predictions, and Stage 1 has not been approved or implemented.

# %% [markdown]
# ## Contents
# 1. Imports
# 2. Notebook-safe configuration lookup
# 3. Fail-closed inference contract

# %%
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import yaml


EXPERIMENT_NAME = "exp401_exp368_weak_risk_candidate_advantage_readout_on_exp264"
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")


# %% [markdown]
# ## 2. Notebook-safe configuration lookup


# %%
def get_nested(mapping: Mapping[str, Any], dotted_key: str) -> Any:
    value: Any = mapping
    for part in dotted_key.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def resolve_package_dir() -> Path:
    cwd = Path.cwd()
    candidates = [
        cwd,
        cwd / "experiments" / EXPERIMENT_NAME,
        KAGGLE_WORKING_ROOT,
    ]
    if KAGGLE_INPUT_ROOT.exists():
        candidates.extend(
            path.parent
            for path in sorted(KAGGLE_INPUT_ROOT.glob("**/config.yaml"))
            if path.parent.name == EXPERIMENT_NAME
        )
    for candidate in candidates:
        config_path = candidate / "config.yaml"
        if not config_path.is_file():
            continue
        loaded = yaml.safe_load(config_path.read_text()) or {}
        if get_nested(loaded, "experiment.name") == EXPERIMENT_NAME:
            return candidate
    raise FileNotFoundError(f"Could not locate config.yaml for {EXPERIMENT_NAME}")


def load_config(package_dir: Path) -> dict[str, Any]:
    value = yaml.safe_load((package_dir / "config.yaml").read_text()) or {}
    if not isinstance(value, dict):
        raise ValueError("config.yaml must contain a mapping")
    return value


# %% [markdown]
# ## 3. Fail-closed inference contract


# %%
def validate_inference_is_disabled(config: Mapping[str, Any]) -> dict[str, Any]:
    status = {
        "experiment": get_nested(config, "experiment.name"),
        "route": get_nested(config, "experiment.route"),
        "experiment_status": get_nested(config, "experiment.status"),
        "implementation_scope": get_nested(config, "implementation.scope"),
        "stage_0_implemented": bool(
            get_nested(config, "implementation.stage_0_implemented")
        ),
        "stage_1_implemented": bool(
            get_nested(config, "implementation.stage_1_implemented")
        ),
        "inference_enabled": bool(
            get_nested(config, "implementation.inference_enabled")
        ),
        "run_inference": bool(get_nested(config, "execution.run_inference")),
        "create_submission": bool(
            get_nested(config, "execution.create_submission")
        ),
    }
    if status["experiment"] != EXPERIMENT_NAME or status["route"] != "ml_model":
        raise ValueError(f"Unexpected exp401 inference contract: {status}")
    enabled = [
        key
        for key in (
            "stage_1_implemented",
            "inference_enabled",
            "run_inference",
            "create_submission",
        )
        if status[key]
    ]
    if enabled:
        raise ValueError(
            "exp401 inference is forbidden before Stage 0 PASS, separate Stage 1 "
            f"approval, and downstream approval: {enabled}"
        )
    return status


if __name__ == "__main__":
    PACKAGE_DIR = resolve_package_dir()
    CONFIG = load_config(PACKAGE_DIR)
    STATUS = validate_inference_is_disabled(CONFIG)
    print(json.dumps(STATUS, indent=2, sort_keys=True))
    raise RuntimeError(
        "exp401 inference is fail-closed: Stage 0 is diagnostic-only and no "
        "downstream inference has been approved."
    )
