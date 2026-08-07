# %% [markdown]
# # exp337 prefix-backtested structure sigma GR inference
#
# exp337 currently implements only the target-free Stage 0 scale/NLL audit.
# Stage 1 HMM prediction, raw-test inference, and submission creation remain
# fail-closed even if Stage 0 later passes.

# %% [markdown]
# ## Contents
# 1. Imports and execution guard
# 2. Notebook-safe configuration helpers
# 3. Disabled inference contract
# 4. Setup and explicit stop

# %%
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml

EXPERIMENT_NAME = "exp337_prefix_backtested_structure_sigma_gr"


def in_notebook_runtime() -> bool:
    try:
        return get_ipython() is not None  # type: ignore[name-defined]
    except NameError:
        return False


EXECUTE_NOTEBOOK = os.environ.get("EXP337_IMPORT_ONLY", "0") != "1" and in_notebook_runtime()


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
    raise FileNotFoundError(f"exp337 config not found in {[str(path) for path in candidates]}")


# %% [markdown]
# ## 3. Disabled inference contract


# %%
def validate_disabled_inference(config: dict[str, Any]) -> dict[str, Any]:
    contract = {
        "experiment": get_nested(config, "experiment.name"),
        "route": get_nested(config, "experiment.route"),
        "active_stage": get_nested(config, "execution.active_stage"),
        "stage0_run": bool(get_nested(config, "execution.run_stage_0")),
        "stage1_enabled": bool(
            get_nested(config, "model.stage_1_exact_hmm.enabled_after_implementation")
        ),
        "run_stage1": bool(get_nested(config, "execution.run_stage_1")),
        "inference_mode": get_nested(config, "inference.mode"),
        "inference_enabled": bool(get_nested(config, "inference.enabled")),
        "run_inference": bool(get_nested(config, "execution.run_inference")),
        "inference_create_submission": bool(get_nested(config, "inference.create_submission")),
        "execution_create_submission": bool(get_nested(config, "execution.create_submission")),
    }
    if contract["experiment"] != EXPERIMENT_NAME or contract["route"] != "pf_beam":
        raise ValueError("unexpected exp337 inference config")
    if contract["inference_mode"] != "disabled_until_stage_1_promotion_and_separate_approval":
        raise ValueError("exp337 inference mode contract changed")
    forbidden_true = (
        "stage1_enabled",
        "run_stage1",
        "inference_enabled",
        "run_inference",
        "inference_create_submission",
        "execution_create_submission",
    )
    if any(contract[key] for key in forbidden_true):
        raise ValueError("exp337 Stage 1, inference, and submission must remain disabled")
    return contract


def stop_disabled_inference(config: dict[str, Any]) -> None:
    contract = validate_disabled_inference(config)
    print(json.dumps(contract, indent=2, sort_keys=True))
    raise RuntimeError(
        "exp337 implements only the train-side Stage 0 scale audit; "
        "Stage 1 inference and submission are disabled"
    )


# %% [markdown]
# ## 4. Setup and explicit stop


# %%
if EXECUTE_NOTEBOOK:
    CONFIG = load_experiment_config()
    stop_disabled_inference(CONFIG)
