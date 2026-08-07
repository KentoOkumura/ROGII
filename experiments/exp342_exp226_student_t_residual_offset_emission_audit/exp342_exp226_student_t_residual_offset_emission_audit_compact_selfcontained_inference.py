# %% [markdown]
# # exp342 exp226 Student-t residual-offset emission audit inference
#
# exp342 implements the Stage 0 rank audit and an explicitly approved Stage 1
# train-side HMM audit. Raw-test regeneration, inference, and submission remain
# fail-closed regardless of the Stage 1 result.

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

EXPERIMENT_NAME = "exp342_exp226_student_t_residual_offset_emission_audit"


def in_notebook_runtime() -> bool:
    try:
        return get_ipython() is not None  # type: ignore[name-defined]
    except NameError:
        return False


EXECUTE_NOTEBOOK = os.environ.get("EXP342_IMPORT_ONLY", "0") != "1" and in_notebook_runtime()


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
    raise FileNotFoundError(f"exp342 config not found in {[str(path) for path in candidates]}")


# %% [markdown]
# ## 3. Disabled inference contract


# %%
def validate_disabled_inference(config: dict[str, Any]) -> dict[str, Any]:
    contract = {
        "experiment": get_nested(config, "experiment.name"),
        "route": get_nested(config, "experiment.route"),
        "active_stage": get_nested(config, "execution.active_stage"),
        "scientific_scores": int(
            get_nested(config, "execution_contract.stage_0.scientific_scores")
        ),
        "control_scores": int(get_nested(config, "execution_contract.stage_0.control_scores")),
        "model_configs": int(get_nested(config, "execution_contract.stage_0.model_configs")),
        "hmm_well_runs": int(get_nested(config, "execution_contract.stage_0.hmm_well_runs")),
        "stage_1_hmm_well_runs": int(
            get_nested(config, "execution_contract.stage_1_override.hmm_well_runs")
        ),
        "stage_1_implemented": bool(get_nested(config, "implementation.stage_1_implemented")),
        "run_inference": bool(get_nested(config, "execution.run_inference")),
        "execution_create_submission": bool(get_nested(config, "execution.create_submission")),
        "inference_enabled": bool(get_nested(config, "inference.enabled")),
        "inference_create_submission": bool(get_nested(config, "inference.create_submission")),
        "implementation_inference_enabled": bool(
            get_nested(config, "implementation.inference_enabled")
        ),
        "implementation_submission_enabled": bool(
            get_nested(config, "implementation.submission_enabled")
        ),
    }
    if contract["experiment"] != EXPERIMENT_NAME or contract["route"] != "pf_beam":
        raise ValueError("unexpected exp342 inference config")
    if contract["scientific_scores"] != 1 or contract["control_scores"] != 1:
        raise ValueError("exp342 Stage 0 must remain one Student-t plus one saved control")
    if contract["model_configs"] != 0:
        raise ValueError("exp342 Stage 0 must remain a zero-model audit")
    if contract["hmm_well_runs"] != 0:
        raise ValueError("exp342 Stage 0 must remain zero-HMM")
    if contract["stage_1_hmm_well_runs"] != 773:
        raise ValueError("exp342 Stage 1 must remain one 773-well HMM variant")
    if not contract["stage_1_implemented"]:
        raise ValueError("exp342 Stage 1 train-side implementation is missing")
    forbidden_true = (
        "run_inference",
        "execution_create_submission",
        "inference_enabled",
        "inference_create_submission",
        "implementation_inference_enabled",
        "implementation_submission_enabled",
    )
    if any(contract[key] for key in forbidden_true):
        raise ValueError("exp342 inference and submission must remain disabled")
    return contract


def stop_disabled_inference(config: dict[str, Any]) -> None:
    contract = validate_disabled_inference(config)
    print(json.dumps(contract, indent=2, sort_keys=True))
    raise RuntimeError(
        "exp342 Stage 1 is train-side only; raw-test inference and submission are disabled"
    )


# %% [markdown]
# ## 4. Setup and explicit stop


# %%
if EXECUTE_NOTEBOOK:
    CONFIG = load_experiment_config()
    stop_disabled_inference(CONFIG)
