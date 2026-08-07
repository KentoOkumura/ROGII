# %% [markdown]
# # exp355 exp226 dip-rate prior on exp209 — disabled inference
#
# exp355 implements only a train-side, zero-HMM Stage 0 identifiability
# readout.  Raw-test schedule generation, HMM decoding, prediction, and
# submission creation remain fail-closed.

# %% [markdown]
# ## Contents
# 1. Imports
# 2. Notebook-safe configuration helpers
# 3. Disabled Stage 1 and inference contract
# 4. Setup and explicit stop

# %% [markdown]
# ## 1. Imports

# %%
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import yaml


EXPERIMENT_NAME = "exp355_exp226_dip_rate_prior_on_exp209"
PACKAGE_DIR = Path.cwd()


# %% [markdown]
# ## 2. Notebook-safe configuration helpers

# %%
def get_nested(config: Mapping[str, Any], dotted_key: str, default: Any = None) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def resolve_config_path() -> Path:
    candidates = [
        PACKAGE_DIR / "config.yaml",
        PACKAGE_DIR / "experiments" / EXPERIMENT_NAME / "config.yaml",
        Path("/kaggle/working/config.yaml"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("exp355 config.yaml was not found")


def read_config() -> dict[str, Any]:
    value = yaml.safe_load(resolve_config_path().read_text()) or {}
    if not isinstance(value, dict):
        raise ValueError("exp355 config must be a YAML mapping")
    return value


# %% [markdown]
# ## 3. Disabled Stage 1 and inference contract

# %%
def validate_disabled_inference(config: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "implementation.stage_0_implemented": True,
        "implementation.stage_1_implemented": True,
        "implementation.inference_enabled": False,
        "implementation.submission_enabled": False,
        "execution.run_inference": False,
        "execution.create_submission": False,
        "inference.enabled": False,
        "inference.create_submission": False,
    }
    for key, expected_value in expected.items():
        if get_nested(config, key) is not expected_value:
            raise ValueError(f"exp355 {key} must remain {expected_value!r}")
    if get_nested(config, "execution_contract.stage_0.hmm_well_runs") != 0:
        raise ValueError("exp355 Stage 0 must remain a zero-HMM readout")
    if get_nested(config, "execution_contract.stage_1_if_pass.hmm_well_runs") != 773:
        raise ValueError("exp355 conditional Stage 1 HMM count changed")
    if get_nested(config, "execution_contract.parent_control_retraining") is not False:
        raise ValueError("exp355 parent/control rerun must remain disabled")
    return {
        "experiment": EXPERIMENT_NAME,
        "stage_0_implemented": True,
        "stage_0_hmm_well_runs": 0,
        "stage_1_implemented": False,
        "conditional_stage_1_hmm_well_runs": 773,
        "canonical_train_notebook_adopted": bool(
            get_nested(config, "implementation.canonical_notebook_adopted")
        ),
        "inference_enabled": False,
        "create_submission": False,
        "parent_control_reruns": 0,
    }


def stop_disabled_inference(config: Mapping[str, Any]) -> None:
    validate_disabled_inference(config)
    raise RuntimeError(
        "exp355 Stage 1, inference, and submission are disabled. "
        "Stage 0 requires separate Kaggle run approval, and a Stage 0 PASS "
        "would still require another approval for 773 exact-HMM well-runs."
    )


# %% [markdown]
# ## 4. Setup and explicit stop

# %%
CONFIG = read_config()
INFERENCE_CONTRACT = validate_disabled_inference(CONFIG)
print("Experiment:", get_nested(CONFIG, "experiment.name"))
print("Route:", get_nested(CONFIG, "experiment.route"))
print("Status:", get_nested(CONFIG, "experiment.status"))
print("Inference contract:", INFERENCE_CONTRACT)

if os.environ.get("EXP355_IMPORT_ONLY") != "1":
    stop_disabled_inference(CONFIG)
