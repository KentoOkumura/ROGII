# %% [markdown]
# # exp284 prefix-masked wrong-mode branch recovery backtest inference
#
# exp284 is a train-side controlled diagnostic. It intentionally has no raw-test
# inference, decoder update, prediction, submission, or selected deployment path.

# %% [markdown]
# ## Contents
# 1. Imports and config helpers
# 2. Disabled-inference contract
# 3. Setup preview
# 4. Fail closed

# %%
from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

EXPERIMENT_NAME = "exp284_prefix_masked_wrong_mode_branch_recovery_backtest"


def in_notebook_runtime() -> bool:
    try:
        return get_ipython() is not None  # type: ignore[name-defined]
    except NameError:
        return False


EXECUTE_NOTEBOOK = os.environ.get("EXP284_IMPORT_ONLY", "0") != "1" and in_notebook_runtime()


# %% [markdown]
# ## 1. Imports and config helpers


# %%
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


def load_config() -> dict[str, Any]:
    root = project_root()
    candidates = (
        Path.cwd() / "config.yaml",
        root / "experiments" / EXPERIMENT_NAME / "config.yaml",
    )
    for path in candidates:
        if not path.exists():
            continue
        value = yaml.safe_load(path.read_text()) or {}
        if get_nested(value, "experiment.name") == EXPERIMENT_NAME:
            return value
    raise FileNotFoundError("exp284 config.yaml was not found")


# %% [markdown]
# ## 2. Disabled-inference contract


# %%
def assert_inference_disabled(config: Mapping[str, Any]) -> None:
    if bool(get_nested(config, "inference.enabled")):
        raise ValueError("exp284 inference.enabled must remain false")
    if bool(get_nested(config, "inference.create_submission")):
        raise ValueError("exp284 create_submission must remain false")
    if bool(get_nested(config, "execution.inference")) or bool(
        get_nested(config, "execution.submission")
    ):
        raise ValueError("exp284 execution inference/submission must remain false")
    if get_nested(config, "inference.mode") != "disabled_backtest_only":
        raise ValueError("exp284 fixes inference.mode=disabled_backtest_only")


def fail_closed() -> None:
    raise RuntimeError(
        "exp284 is a train-side prefix-masked recovery backtest; raw-test inference, "
        "decoder updates, predictions, and submissions are intentionally disabled"
    )


# %% [markdown]
# ## 3. Setup preview

# %%
CONFIG: dict[str, Any] | None = None
if EXECUTE_NOTEBOOK:
    CONFIG = load_config()
    assert_inference_disabled(CONFIG)
    print(
        json.dumps(
            {
                "experiment": get_nested(CONFIG, "experiment.name"),
                "route": get_nested(CONFIG, "experiment.route"),
                "mode": get_nested(CONFIG, "inference.mode"),
                "enabled": get_nested(CONFIG, "inference.enabled"),
                "create_submission": get_nested(CONFIG, "inference.create_submission"),
            },
            indent=2,
        )
    )


# %% [markdown]
# ## 4. Fail closed

# %%
if EXECUTE_NOTEBOOK:
    fail_closed()
