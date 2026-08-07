# %% [markdown]
# # exp402 fold-safe GRWR-5 add-only on exp287 — inference
#
# Inference is intentionally unavailable. Stage 0 passed and the approved
# 15-booster downstream training stage is implemented, but promotion evidence
# does not exist until that run finishes. This candidate therefore remains
# fail-closed for inference and submission.

# %% [markdown]
# ## Contents
#
# 1. Imports
# 2. Notebook-safe configuration lookup
# 3. Fail-closed inference contract

# %%
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


EXPERIMENT_NAME = "exp402_fold_safe_grwr_5_addonly_on_exp287"
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
    raise FileNotFoundError(f"could not locate config for {EXPERIMENT_NAME}")


def load_config(package_dir: Path) -> dict[str, Any]:
    value = yaml.safe_load((package_dir / "config.yaml").read_text()) or {}
    if not isinstance(value, dict):
        raise ValueError("config.yaml must contain a mapping")
    return value


# %% [markdown]
# ## 3. Fail-closed inference contract

# %%
def validate_inference_is_disabled(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    status = {
        "experiment": get_nested(config, "experiment.name"),
        "route": get_nested(config, "experiment.route"),
        "experiment_status": get_nested(config, "experiment.status"),
        "implementation_scope": get_nested(config, "implementation.scope"),
        "stage_0_implemented": bool(
            get_nested(config, "implementation.preflight_implemented")
        ),
        "stage_1_training_implemented": bool(
            get_nested(config, "implementation.training_implemented")
        ),
        "inference_implemented": bool(
            get_nested(config, "implementation.inference_implemented")
        ),
        "submission_enabled": bool(
            get_nested(config, "implementation.submission_enabled")
        ),
        "run_train": bool(get_nested(config, "execution.run_train")),
        "run_inference": bool(get_nested(config, "execution.run_inference")),
        "create_submission": bool(
            get_nested(config, "execution.create_submission")
        ),
    }
    if status["experiment"] != EXPERIMENT_NAME or status["route"] != "ml_model":
        raise ValueError(f"unexpected exp402 inference contract: {status}")
    enabled = [
        key
        for key in (
            "inference_implemented",
            "submission_enabled",
            "run_inference",
            "create_submission",
        )
        if status[key]
    ]
    if enabled:
        raise ValueError(
            "exp402 inference is forbidden before Stage 1 promotion PASS and "
            f"separate inference approval: {enabled}"
        )
    if not status["stage_0_implemented"] or not status[
        "stage_1_training_implemented"
    ]:
        raise ValueError("exp402 Stage 0/Stage 1 implementation state is incomplete")
    return status


if __name__ == "__main__":
    PACKAGE_DIR = resolve_package_dir()
    CONFIG = load_config(PACKAGE_DIR)
    STATUS = validate_inference_is_disabled(CONFIG)
    print(json.dumps(STATUS, indent=2, sort_keys=True))
    raise RuntimeError(
        "exp402 inference is fail-closed until the Stage 1 GPU run passes every "
        "promotion gate and receives separate inference approval."
    )
