# %% [markdown]
# # exp368 marginalized reliability PF — inference
#
# exp368 currently implements only the zero-PF train-side Stage 0 reliability
# readout. Stage 1 PF replay, raw-test inference, and submission are fail-closed.

# %% [markdown]
# ## Contents
# 1. Imports and notebook-safe configuration
# 2. Disabled inference contract
# 3. Setup and explicit stop

# %%
from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

try:
    from IPython import get_ipython
    from IPython.display import display
except ImportError:  # pragma: no cover
    def get_ipython() -> None:
        return None

    def display(value: Any) -> None:
        print(value)


EXPERIMENT_NAME = "exp368_marginalized_reliability_pf"
PACKAGE_DIR = Path.cwd()
IMPORT_ONLY = os.environ.get("EXP368_IMPORT_ONLY", "0") == "1"
EXECUTE_NOTEBOOK = get_ipython() is not None and not IMPORT_ONLY


def read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text()) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def get_nested(config: Mapping[str, Any], dotted_key: str) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def project_root() -> Path:
    for candidate in (PACKAGE_DIR, *PACKAGE_DIR.parents):
        if (candidate / "project.yml").is_file():
            return candidate
    return PACKAGE_DIR


def load_config() -> dict[str, Any]:
    root = project_root()
    candidates = (
        PACKAGE_DIR / "config.yaml",
        root / "experiments" / EXPERIMENT_NAME / "config.yaml",
    )
    for path in candidates:
        if not path.is_file():
            continue
        config = read_yaml(path)
        if get_nested(config, "experiment.name") == EXPERIMENT_NAME:
            return config
    raise FileNotFoundError(f"Could not locate exp368 config; checked={candidates}")


CONFIG = load_config()

# %% [markdown]
# ## 2. Disabled inference contract


# %%
def validate_disabled_inference(config: Mapping[str, Any]) -> dict[str, Any]:
    checks = {
        "experiment.route": "pf_beam",
        "implementation.enabled": True,
        "implementation.stage_1_implemented": False,
        "execution.run_stage_1": False,
        "execution.run_inference": False,
        "execution.create_submission": False,
        "inference.enabled": False,
        "inference.create_submission": False,
    }
    for key, expected in checks.items():
        actual = get_nested(config, key)
        if actual != expected:
            raise ValueError(
                f"exp368 disabled inference contract changed: "
                f"{key}={actual!r}, expected {expected!r}"
            )
    return dict(get_nested(config, "execution_contract.stage_0") or {})


def stop_disabled_inference(config: Mapping[str, Any]) -> None:
    validate_disabled_inference(config)
    raise RuntimeError(
        "exp368 implements only the target-free Stage 0 reliability readout. "
        "Stage 1 PF replay, inference, and submission require separate approval."
    )


# %% [markdown]
# ## 3. Setup and explicit stop


# %%
if EXECUTE_NOTEBOOK:
    display(
        {
            "experiment": EXPERIMENT_NAME,
            "route": get_nested(CONFIG, "experiment.route"),
            "status": get_nested(CONFIG, "experiment.status"),
            "stage_0_execution_contract": validate_disabled_inference(CONFIG),
            "stage_1_implemented": False,
            "inference_enabled": False,
            "submission_enabled": False,
        }
    )
    stop_disabled_inference(CONFIG)
