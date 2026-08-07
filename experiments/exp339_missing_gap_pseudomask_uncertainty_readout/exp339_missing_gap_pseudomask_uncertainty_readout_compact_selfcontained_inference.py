# %% [markdown]
# # exp339 missing-gap pseudo-mask uncertainty readout inference
#
# exp339 implements only a train-side known-prefix Stage 0 uncertainty
# readout. It creates no TVT prediction. HMM integration, raw-test inference,
# and submission creation remain fail-closed even if every Stage 0 gate later
# passes.

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

EXPERIMENT_NAME = "exp339_missing_gap_pseudomask_uncertainty_readout"


def in_notebook_runtime() -> bool:
    try:
        return get_ipython() is not None  # type: ignore[name-defined]
    except NameError:
        return False


EXECUTE_NOTEBOOK = os.environ.get("EXP339_IMPORT_ONLY", "0") != "1" and in_notebook_runtime()


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
    raise FileNotFoundError(f"exp339 config not found in {[str(path) for path in candidates]}")


# %% [markdown]
# ## 3. Disabled inference contract


# %%
def validate_disabled_inference(config: dict[str, Any]) -> dict[str, Any]:
    contract = {
        "experiment": get_nested(config, "experiment.name"),
        "route": get_nested(config, "experiment.route"),
        "active_stage": get_nested(config, "execution.active_stage"),
        "stage0_completed": bool(get_nested(config, "execution.stage_0_completed")),
        "stage0_gate_passed": get_nested(config, "execution.stage_0_gate_passed"),
        "hmm_well_runs": int(get_nested(config, "execution_contract.hmm_well_runs")),
        "run_inference": bool(get_nested(config, "execution.run_inference")),
        "inference_mode": get_nested(config, "inference.mode"),
        "inference_enabled": bool(get_nested(config, "inference.enabled")),
        "inference_create_submission": bool(get_nested(config, "inference.create_submission")),
        "execution_create_submission": bool(get_nested(config, "execution.create_submission")),
    }
    if contract["experiment"] != EXPERIMENT_NAME or contract["route"] != "pf_beam":
        raise ValueError("unexpected exp339 inference config")
    if contract["inference_mode"] != (
        "disabled_until_all_stage0_gates_pass_and_separate_implementation_approval"
    ):
        raise ValueError("exp339 inference mode contract changed")
    if contract["hmm_well_runs"] != 0:
        raise ValueError("exp339 must not run an HMM")
    forbidden_true = (
        "run_inference",
        "inference_enabled",
        "inference_create_submission",
        "execution_create_submission",
    )
    if any(contract[key] for key in forbidden_true):
        raise ValueError("exp339 inference and submission must remain disabled")
    return contract


def stop_disabled_inference(config: dict[str, Any]) -> None:
    contract = validate_disabled_inference(config)
    print(json.dumps(contract, indent=2, sort_keys=True))
    raise RuntimeError(
        "exp339 implements only the train-side Stage 0 interpolation-uncertainty readout; "
        "HMM integration, inference, and submission are disabled"
    )


# %% [markdown]
# ## 4. Setup and explicit stop


# %%
if EXECUTE_NOTEBOOK:
    CONFIG = load_experiment_config()
    stop_disabled_inference(CONFIG)
