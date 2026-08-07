# %% [markdown]
# # exp285 exp226 prefix-masked offset predictability readout inference
#
# This experiment is a train-side diagnostic only. The inference notebook is
# deliberately fail-closed and cannot create a prediction or submission.

# %% [markdown]
# ## Contents
# 1. Imports and config helpers
# 2. Disabled-inference contract
# 3. Setup preview
# 4. Fail closed

# %% [markdown]
# ## 1. Imports and config helpers

# %%
from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

EXPERIMENT_NAME = "exp285_exp226_prefix_masked_offset_predictability_readout"


def in_notebook_runtime() -> bool:
    try:
        return get_ipython() is not None  # type: ignore[name-defined]
    except NameError:
        return False


EXECUTE_NOTEBOOK = os.environ.get("EXP285_IMPORT_ONLY", "0") != "1" and in_notebook_runtime()


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = yaml.safe_load(path.read_text()) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def get_nested(config: Mapping[str, Any], dotted_key: str, default: Any = None) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def project_root() -> Path:
    start = Path.cwd()
    for candidate in (start, *start.parents):
        if (candidate / "project.yml").exists():
            return candidate
    return start


def load_experiment_config() -> dict[str, Any]:
    root = project_root()
    candidates = (
        Path.cwd() / "config.yaml",
        root / "experiments" / EXPERIMENT_NAME / "config.yaml",
    )
    for path in candidates:
        config = read_yaml(path)
        if get_nested(config, "experiment.name") == EXPERIMENT_NAME:
            return config
    raise FileNotFoundError(f"exp285 config not found in {[str(path) for path in candidates]}")


# %% [markdown]
# ## 2. Disabled-inference contract


# %%
def assert_inference_disabled(config: Mapping[str, Any]) -> None:
    if get_nested(config, "experiment.route") != "pf_beam":
        raise ValueError("exp285 fixes route=pf_beam")
    if bool(get_nested(config, "execution.inference")):
        raise ValueError("exp285 execution.inference must remain false")
    if bool(get_nested(config, "execution.submission")):
        raise ValueError("exp285 execution.submission must remain false")
    if bool(get_nested(config, "inference.enabled")):
        raise ValueError("exp285 inference.enabled must remain false")
    if bool(get_nested(config, "inference.create_submission")):
        raise ValueError("exp285 inference.create_submission must remain false")


def fail_closed() -> None:
    raise RuntimeError(
        "exp285 is a train-side prefix-masked offset predictability readout; "
        "prediction, current-test generation, inference, and submission are disabled."
    )


# %% [markdown]
# ## 3. Setup preview


# %%
CONFIG: dict[str, Any] | None = None
if EXECUTE_NOTEBOOK:
    CONFIG = load_experiment_config()
    assert_inference_disabled(CONFIG)
    print(
        json.dumps(
            {
                "experiment": get_nested(CONFIG, "experiment.name"),
                "route": get_nested(CONFIG, "experiment.route"),
                "status": get_nested(CONFIG, "experiment.status"),
                "inference_enabled": get_nested(CONFIG, "inference.enabled"),
                "create_submission": get_nested(CONFIG, "inference.create_submission"),
                "message": "train-side readout only; fail closed",
            },
            indent=2,
        )
    )


# %% [markdown]
# ## 4. Fail closed


# %%
if EXECUTE_NOTEBOOK:
    fail_closed()
