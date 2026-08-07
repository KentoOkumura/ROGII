# %% [markdown]
# # exp280 exp226 shift-likelihood separability readout inference
#
# This experiment is a train-side diagnostic only. The inference notebook is
# deliberately fail-closed and cannot create a prediction or submission.

# %% [markdown]
# ## Contents
# 1. Imports and experiment contract
# 2. Notebook-safe configuration helpers
# 3. Disabled inference validation
# 4. Setup and explicit stop

# %%
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml


EXPERIMENT_NAME = "exp280_exp226_shift_likelihood_separability_readout"


def in_notebook_runtime() -> bool:
    try:
        return get_ipython() is not None  # type: ignore[name-defined]
    except NameError:
        return False


EXECUTE_NOTEBOOK = os.environ.get("EXP280_IMPORT_ONLY", "0") != "1" and in_notebook_runtime()


# %% [markdown]
# ## 2. Notebook-safe configuration helpers


# %%
def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = yaml.safe_load(path.read_text()) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def get_nested(config: dict[str, Any], dotted_key: str) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
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
    raise FileNotFoundError(f"exp280 config not found in {[str(path) for path in candidates]}")


# %% [markdown]
# ## 3. Disabled inference validation


# %%
def validate_disabled_inference(config: dict[str, Any]) -> dict[str, Any]:
    contract = {
        "experiment": get_nested(config, "experiment.name"),
        "route": get_nested(config, "experiment.route"),
        "inference_enabled": bool(get_nested(config, "inference.enabled")),
        "create_submission": bool(get_nested(config, "inference.create_submission")),
        "execution_inference": bool(get_nested(config, "execution.inference")),
        "execution_submission": bool(get_nested(config, "execution.submission")),
        "mode": get_nested(config, "inference.mode"),
    }
    if contract["experiment"] != EXPERIMENT_NAME:
        raise ValueError("unexpected experiment config")
    if any(
        contract[key]
        for key in (
            "inference_enabled",
            "create_submission",
            "execution_inference",
            "execution_submission",
        )
    ):
        raise ValueError("exp280 inference and submission must remain disabled")
    if contract["mode"] != "disabled_train_side_separability_readout_only":
        raise ValueError("exp280 inference mode contract changed")
    return contract


def stop_disabled_inference(config: dict[str, Any]) -> None:
    contract = validate_disabled_inference(config)
    print(json.dumps(contract, indent=2, sort_keys=True))
    raise RuntimeError(
        "exp280 is a train-side separability readout only; inference and submission are disabled"
    )


# %% [markdown]
# ## 4. Setup and explicit stop


# %%
if EXECUTE_NOTEBOOK:
    CONFIG = load_experiment_config()
    stop_disabled_inference(CONFIG)

