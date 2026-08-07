# %% [markdown]
# # exp358 exp209 missing-distance emission downweight inference
#
# Inference is intentionally unavailable. Exp358 implements the train-side
# Stage 0 audit and separately approved Stage 1 exact-HMM evaluation only.
# Train-side completion does not authorize raw-test generation, inference,
# or submission.

# %% [markdown]
# ## Contents
# 1. Imports and execution guard
# 2. Configuration preview
# 3. Fail-closed inference boundary

# %%
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml

EXPERIMENT_NAME = "exp358_exp209_missing_distance_emission_downweight"


def in_notebook_runtime() -> bool:
    try:
        return get_ipython() is not None  # type: ignore[name-defined]
    except NameError:
        return False


EXECUTE_NOTEBOOK = os.environ.get("EXP358_IMPORT_ONLY", "0") != "1" and in_notebook_runtime()


# %% [markdown]
# ## 2. Configuration preview


# %%
def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = yaml.safe_load(path.read_text()) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


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
        if (config.get("experiment") or {}).get("name") == EXPERIMENT_NAME:
            return config
    raise FileNotFoundError("exp358 config not found")


def validate_inference_disabled(config: dict[str, Any]) -> None:
    inference = config.get("inference") or {}
    execution = config.get("execution") or {}
    if inference.get("enabled") is not False:
        raise ValueError("exp358 inference must remain disabled")
    if inference.get("create_submission") is not False:
        raise ValueError("exp358 submission creation must remain disabled")
    if execution.get("run_inference") is not False:
        raise ValueError("exp358 execution.run_inference must remain false")
    if execution.get("create_submission") is not False:
        raise ValueError("exp358 execution.create_submission must remain false")


# %% [markdown]
# ## 3. Fail-closed inference boundary


# %%
def run_inference(_: dict[str, Any]) -> None:
    raise RuntimeError(
        "exp358 inference is not implemented or approved; "
        "Stage 1 train-side execution still requires separate inference approval"
    )


if EXECUTE_NOTEBOOK:
    CONFIG = load_experiment_config()
    validate_inference_disabled(CONFIG)
    print(
        json.dumps(
            {
                "experiment": EXPERIMENT_NAME,
                "status": (CONFIG.get("experiment") or {}).get("status"),
                "train_side_only": True,
                "stage_1_implemented": True,
                "inference_enabled": False,
                "submission_enabled": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    run_inference(CONFIG)
