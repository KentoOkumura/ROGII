# %% [markdown]
# # exp306 robust RTS / L1 convergence calibration audit inference
#
# exp306 is a train-side, target-free solver audit. Prediction, raw-test
# inference, and submission generation are outside its frozen scope.

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

EXPERIMENT_NAME = "exp306_robust_rts_l1_convergence_calibration_audit"


def in_notebook_runtime() -> bool:
    try:
        return get_ipython() is not None  # type: ignore[name-defined]
    except NameError:
        return False


EXECUTE_NOTEBOOK = (
    os.environ.get("EXP306_IMPORT_ONLY", "0") != "1" and in_notebook_runtime()
)


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
    raise FileNotFoundError(f"exp306 config not found in {[str(path) for path in candidates]}")


# %% [markdown]
# ## 3. Disabled inference contract


# %%
def validate_disabled_inference(config: dict[str, Any]) -> dict[str, Any]:
    contract = {
        "experiment": get_nested(config, "experiment.name"),
        "route": get_nested(config, "experiment.route"),
        "mode": get_nested(config, "inference.mode"),
        "inference_enabled": bool(get_nested(config, "inference.enabled")),
        "run_inference": bool(get_nested(config, "execution.run_inference")),
        "inference_create_submission": bool(get_nested(config, "inference.create_submission")),
        "execution_create_submission": bool(get_nested(config, "execution.create_submission")),
        "run_scientific_score": bool(get_nested(config, "execution.run_scientific_score")),
    }
    if contract["experiment"] != EXPERIMENT_NAME or contract["route"] != "pf_beam":
        raise ValueError("unexpected exp306 inference config")
    if contract["mode"] != "disabled_train_side_solver_audit_only":
        raise ValueError("exp306 inference mode contract changed")
    enabled = [
        key
        for key in (
            "inference_enabled",
            "run_inference",
            "inference_create_submission",
            "execution_create_submission",
            "run_scientific_score",
        )
        if contract[key]
    ]
    if enabled:
        raise ValueError(
            f"exp306 scientific scoring, inference, and submission must remain disabled: {enabled}"
        )
    return contract


def stop_disabled_inference(config: dict[str, Any]) -> None:
    contract = validate_disabled_inference(config)
    print(json.dumps(contract, indent=2, sort_keys=True))
    raise RuntimeError(
        "exp306 is a train-side target-free solver convergence audit; "
        "prediction, inference, and submission are disabled"
    )


# %% [markdown]
# ## 4. Setup and explicit stop


# %%
if EXECUTE_NOTEBOOK:
    CONFIG = load_experiment_config()
    stop_disabled_inference(CONFIG)
