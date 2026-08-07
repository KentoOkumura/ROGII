# %% [markdown]
# # exp366 fault-reset duration semi-Markov HMM — fail-closed inference
#
# Stage 0 is a train-side, zero-HMM diagnostic. This notebook deliberately
# cannot generate test predictions or a submission. A future Stage 1 requires
# every Stage 0 gate to pass and a separate user approval.

# %% [markdown]
# ## Contents
# 1. Imports and notebook-safe config loading
# 2. Disabled inference contract
# 3. Fail-closed execution boundary

# %% [markdown]
# ## 1. Imports and notebook-safe config loading

# %%
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

try:
    from IPython.display import display
except ImportError:

    def display(value: Any) -> None:
        print(value)


EXPERIMENT_NAME = "exp366_fault_reset_duration_semimarkov_hmm"
PACKAGE_DIR = Path.cwd()


def get_nested(mapping: Mapping[str, Any], dotted_key: str) -> Any:
    current: Any = mapping
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def locate_config() -> Path:
    relative = Path("experiments") / EXPERIMENT_NAME / "config.yaml"
    for candidate in (
        PACKAGE_DIR / "config.yaml",
        PACKAGE_DIR / relative,
        Path.cwd() / "config.yaml",
        Path.cwd() / relative,
        Path("/kaggle/working/config.yaml"),
    ):
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not locate exp366 config.yaml")


def load_config() -> dict[str, Any]:
    value = yaml.safe_load(locate_config().read_text()) or {}
    if not isinstance(value, dict):
        raise ValueError("exp366 config must be a YAML mapping")
    return value


# %% [markdown]
# ## 2. Disabled inference contract

# %%
def validate_disabled_inference(config: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "experiment.name": EXPERIMENT_NAME,
        "experiment.route": "pf_beam",
        "implementation.stage_0_implemented": True,
        "implementation.stage_1_implemented": False,
        "implementation.inference_enabled": False,
        "implementation.submission_enabled": False,
        "execution.run_stage_1": False,
        "execution.run_inference": False,
        "execution.create_submission": False,
        "inference.enabled": False,
        "inference.create_submission": False,
    }
    mismatches = {
        key: {"expected": expected, "actual": get_nested(config, key)}
        for key, expected in required.items()
        if get_nested(config, key) != expected
    }
    if mismatches:
        raise ValueError(f"exp366 disabled inference contract mismatch: {mismatches}")
    counts = get_nested(config, "execution.stage_0_counts") or {}
    if counts != {
        "diagnostic_variants": 1,
        "fixed_branches": 13,
        "reporting_folds": 5,
        "semimarkov_hmm_well_runs": 0,
        "lightgbm_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
        "parent_control_reruns": 0,
    }:
        raise ValueError("exp366 Stage 0 execution counts changed")
    return counts


def stop_disabled_inference(config: Mapping[str, Any]) -> None:
    validate_disabled_inference(config)
    raise RuntimeError(
        "exp366 inference is disabled: Stage 1 semi-Markov HMM is not "
        "implemented or approved. Run the fixed Stage 0 preflight first; "
        "all gates and a separate approval are required before inference."
    )


# %% [markdown]
# ## 3. Fail-closed execution boundary

# %%
CONFIG = load_config()
COUNTS = validate_disabled_inference(CONFIG)
display(
    pd.DataFrame(
        [
            {
                "experiment": EXPERIMENT_NAME,
                "route": get_nested(CONFIG, "experiment.route"),
                "status": get_nested(CONFIG, "experiment.status"),
                "stage_0_implemented": get_nested(
                    CONFIG, "implementation.stage_0_implemented"
                ),
                "stage_1_implemented": get_nested(
                    CONFIG, "implementation.stage_1_implemented"
                ),
                "fixed_branches": COUNTS["fixed_branches"],
                "semimarkov_hmm_well_runs": COUNTS[
                    "semimarkov_hmm_well_runs"
                ],
                "inference_enabled": get_nested(CONFIG, "inference.enabled"),
                "create_submission": get_nested(
                    CONFIG, "inference.create_submission"
                ),
            }
        ]
    )
)
print(
    "Fail closed: this Stage 0 implementation does not create predictions or "
    "submission.csv."
)
