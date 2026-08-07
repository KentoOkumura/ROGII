# %% [markdown]
# # exp430 Huber seed-evidence reaggregation inference
#
# Inference is intentionally disabled. exp430 is a train-side OOF PF audit;
# raw-test replay and submission require the train merge to pass every
# scientific and tail gate and then require separate user approval.

# %% [markdown]
# ## Contents
# 1. Imports
# 2. Runtime and configuration helpers
# 3. Setup and frozen train-side requirements
# 4. Disabled inference guard

# %%
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

import yaml

EXPERIMENT_NAME = "exp430_huber_seed_evidence_reaggregation"
KAGGLE_INPUT_ROOT = Path("/kaggle/input")


def in_notebook_runtime() -> bool:
    try:
        return get_ipython() is not None  # type: ignore[name-defined]
    except NameError:
        return False


EXECUTE_NOTEBOOK = os.environ.get("EXP430_IMPORT_ONLY", "0") != "1" and in_notebook_runtime()


# %% [markdown]
# ## 2. Runtime and configuration helpers


# %%
def get_nested(config: Mapping[str, Any], dotted_key: str) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


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
    candidates = [
        Path.cwd() / "config.yaml",
        root / "experiments" / EXPERIMENT_NAME / "config.yaml",
        Path("/kaggle/working/config.yaml"),
    ]
    if KAGGLE_INPUT_ROOT.exists():
        candidates.extend(sorted(KAGGLE_INPUT_ROOT.glob("**/config.yaml")))
    for path in candidates:
        config = read_yaml(path)
        if get_nested(config, "experiment.name") == EXPERIMENT_NAME:
            return config
    raise FileNotFoundError("exp430 config.yaml was not found")


def validate_inference_disabled(config: Mapping[str, Any]) -> dict[str, Any]:
    checks = {
        "route": get_nested(config, "experiment.route") == "pf_beam",
        "inference_enabled": get_nested(config, "inference.enabled") is False,
        "execution_inference_approved": (
            get_nested(config, "execution.inference_approved") is False
        ),
        "submission_approved": (
            get_nested(config, "execution.submission_approved") is False
        ),
        "selected_candidate": get_nested(config, "inference.selected_candidate")
        is None,
    }
    if not all(checks.values()):
        raise RuntimeError(
            "exp430 inference contract changed; create a separately approved "
            "same-experiment inference design before executing"
        )
    return {
        "experiment": EXPERIMENT_NAME,
        "status": "inference_disabled_pending_train_gate_and_separate_approval",
        "checks": checks,
    }


# %% [markdown]
# ## 3. Setup and frozen train-side requirements


# %%
if EXECUTE_NOTEBOOK:
    CONFIG = load_experiment_config()
    INFERENCE_CONTRACT = validate_inference_disabled(CONFIG)
    print(
        json.dumps(
            {
                **INFERENCE_CONTRACT,
                "required_train_status": (
                    "train_side_huber_seed_evidence_gate_passed_no_automatic_inference"
                ),
                "required_train_artifacts": [
                    "promotion_gate.json",
                    "combined_prediction_identity.json",
                    "summary.json",
                ],
                "raw_test_pf_replays_currently_approved": 0,
                "models": 0,
                "boosters": 0,
            },
            indent=2,
            sort_keys=True,
        )
    )


# %% [markdown]
# ## 4. Disabled inference guard


# %%
if EXECUTE_NOTEBOOK:
    raise RuntimeError(
        "exp430 inference is disabled. A passing train-side merge does not "
        "automatically authorize raw-test PF replay, submission creation, or "
        "Kaggle submission."
    )
