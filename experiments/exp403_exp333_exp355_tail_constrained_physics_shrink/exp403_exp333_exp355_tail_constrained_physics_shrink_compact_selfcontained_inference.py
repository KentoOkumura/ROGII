# %% [markdown]
# # exp403 exp333/exp355 tail-constrained physics shrink — inference
#
# Current-test inference is intentionally unavailable.  The saved-OOF
# train-side policy must first pass every promotion gate, and a separate
# implementation/run approval must then freeze the three-well current-test
# regeneration contract.

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

EXPERIMENT_NAME = "exp403_exp333_exp355_tail_constrained_physics_shrink"
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")


# %% [markdown]
# ## 2. Notebook-safe configuration lookup

# %%
def get_nested(mapping: Mapping[str, Any], dotted_key: str, default: Any = None) -> Any:
    value: Any = mapping
    for part in dotted_key.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return default
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
def validate_inference_is_disabled(config: Mapping[str, Any]) -> dict[str, Any]:
    status = {
        "experiment": get_nested(config, "experiment.name"),
        "route": get_nested(config, "experiment.route"),
        "experiment_status": get_nested(config, "experiment.status"),
        "train_readout_implemented": bool(
            get_nested(config, "implementation.train_readout_implemented")
        ),
        "canonical_notebook_adopted": bool(
            get_nested(config, "implementation.canonical_notebook_adopted")
        ),
        "training_enabled": bool(
            get_nested(config, "implementation.training_enabled")
        ),
        "inference_enabled": bool(
            get_nested(config, "implementation.inference_enabled")
        ),
        "submission_enabled": bool(
            get_nested(config, "implementation.submission_enabled")
        ),
        "run_train": bool(get_nested(config, "execution.run_train")),
        "run_inference": bool(get_nested(config, "execution.run_inference")),
        "create_submission": bool(
            get_nested(config, "execution.create_submission")
        ),
        "promotion_result": get_nested(config, "results.promotion_gate_passed"),
    }
    if status["experiment"] != EXPERIMENT_NAME or status["route"] != "ensemble":
        raise ValueError(f"unexpected exp403 inference contract: {status}")
    enabled = [
        key
        for key in (
            "inference_enabled",
            "submission_enabled",
            "run_inference",
            "create_submission",
        )
        if status[key]
    ]
    if enabled:
        raise ValueError(
            "exp403 inference is forbidden before train-side promotion PASS and "
            f"separate current-test approval: {enabled}"
        )
    if status["promotion_result"] is True:
        raise ValueError(
            "promotion PASS must be reviewed and explicitly approved before this "
            "fail-closed candidate is replaced"
        )
    return status


if __name__ == "__main__":
    PACKAGE_DIR = resolve_package_dir()
    CONFIG = load_config(PACKAGE_DIR)
    STATUS = validate_inference_is_disabled(CONFIG)
    print(json.dumps(STATUS, indent=2, sort_keys=True))
    raise RuntimeError(
        "exp403 inference is fail-closed: only the saved-OOF train-side "
        "implementation candidate exists."
    )
