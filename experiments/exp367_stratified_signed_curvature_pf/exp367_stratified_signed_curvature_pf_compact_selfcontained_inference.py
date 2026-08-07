# %% [markdown]
# # exp367_stratified_signed_curvature_pf inference
#
# Inference is intentionally unavailable. Stage 1 PF has not been implemented
# and requires both a Stage 0 PASS and separate user approval.

# %% [markdown]
# ## Contents
# 1. Imports
# 2. Notebook-safe configuration lookup
# 3. Fail-closed inference contract

# %%
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

EXPERIMENT_NAME = "exp367_stratified_signed_curvature_pf"
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")


# %% [markdown]
# ## 2. Notebook-safe configuration lookup

# %%
def get_nested(mapping: dict[str, Any], dotted_key: str) -> Any:
    value: Any = mapping
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
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
        if not config_path.exists():
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
#
# No sample submission is copied and no prediction file is created.

# %%
def validate_inference_is_disabled(config: dict[str, Any]) -> dict[str, Any]:
    status = {
        "experiment": get_nested(config, "experiment.name"),
        "route": get_nested(config, "experiment.route"),
        "experiment_status": get_nested(config, "experiment.status"),
        "implementation_scope": get_nested(config, "execution.implementation_scope"),
        "run_stage_1": bool(get_nested(config, "execution.run_stage_1")),
        "run_inference": bool(get_nested(config, "execution.run_inference")),
        "create_submission": bool(get_nested(config, "execution.create_submission")),
    }
    if status["experiment"] != EXPERIMENT_NAME or status["route"] != "pf_beam":
        raise ValueError(f"Unexpected exp367 inference contract: {status}")
    forbidden_enabled = [
        key
        for key in ("run_stage_1", "run_inference", "create_submission")
        if status[key]
    ]
    if forbidden_enabled:
        raise ValueError(
            "Inference flags cannot be enabled before Stage 1 PF implementation: "
            f"{forbidden_enabled}"
        )
    return status


if __name__ == "__main__":
    PACKAGE_DIR = resolve_package_dir()
    CONFIG = load_config(PACKAGE_DIR)
    STATUS = validate_inference_is_disabled(CONFIG)
    print(json.dumps(STATUS, indent=2))
    raise RuntimeError(
        "exp367 inference is fail-closed: Stage 1 PF is not implemented. "
        "A Stage 0 PASS and separate implementation approval are required."
    )
