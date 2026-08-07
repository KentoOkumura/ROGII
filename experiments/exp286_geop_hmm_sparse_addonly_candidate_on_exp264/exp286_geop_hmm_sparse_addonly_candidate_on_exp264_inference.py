# %% [markdown]
# # exp286 geop HMM sparse add-only candidate on exp264 inference
#
# Inference is intentionally disabled. Stage 0 is a saved-OOF candidate-headroom
# readout and does not authorize hidden-test geop generation or submission.

# %% [markdown]
# ## Contents
# 1. Imports
# 2. Notebook-safe configuration helpers
# 3. Fail-closed staged dependency validation
# 4. Setup and explicit stop

# %%
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import yaml


# %% [markdown]
# ## 2. Notebook-safe configuration helpers

# %%
EXPERIMENT_NAME = "exp286_geop_hmm_sparse_addonly_candidate_on_exp264"
PACKAGE_DIR = Path.cwd()


def in_notebook_runtime() -> bool:
    try:
        shell = get_ipython()  # type: ignore[name-defined]
    except NameError:
        return False
    return shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell"


EXECUTE_NOTEBOOK = (
    os.environ.get("EXP286_IMPORT_ONLY", "0") != "1" and in_notebook_runtime()
)


def get_nested(mapping: Mapping[str, Any], dotted_key: str) -> Any:
    current: Any = mapping
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def find_config_path() -> Path:
    candidates = [
        PACKAGE_DIR / "config.yaml",
        PACKAGE_DIR / "experiments" / EXPERIMENT_NAME / "config.yaml",
    ]
    existing = [path for path in candidates if path.exists()]
    if len(existing) == 1:
        return existing[0]
    matches = sorted(PACKAGE_DIR.rglob(f"{EXPERIMENT_NAME}/config.yaml"))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError("exp286 config.yaml was not found unambiguously")


def read_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text()) or {}
    if not isinstance(value, dict):
        raise ValueError("config must contain a YAML mapping")
    return value


# %% [markdown]
# ## 3. Fail-closed staged dependency validation

# %%
def validate_disabled_inference(config: Mapping[str, Any]) -> dict[str, Any]:
    required_false = [
        "execution.stage_b_enabled",
        "execution.stage_c_enabled",
        "execution.stage_d_enabled",
        "execution.exp276_corrected_guard_passed",
        "execution.inference",
        "execution.submission",
        "inference.enabled",
        "inference.create_submission",
    ]
    unexpected = {
        key: get_nested(config, key)
        for key in required_false
        if get_nested(config, key) is not False
    }
    if unexpected:
        raise ValueError(f"exp286 disabled inference contract changed: {unexpected}")
    expected_zero = [
        "execution.lightgbm_config_count",
        "execution.trained_fold_count",
        "execution.total_boosters",
        "execution.hmm_well_runs",
        "execution.pf_well_runs",
    ]
    nonzero = {
        key: get_nested(config, key)
        for key in expected_zero
        if get_nested(config, key) != 0
    }
    if nonzero:
        raise ValueError(f"exp286 inference has unauthorized execution: {nonzero}")
    return {
        "experiment": EXPERIMENT_NAME,
        "status": "disabled",
        "reason": get_nested(config, "inference.mode"),
        "stage0_is_saved_oof_readout_only": True,
        "required_before_inference": [
            "Stage 0 candidate guard PASS",
            "exp276 corrected revalidation all guards PASS",
            "Stage B 10 CPU models separately approved and passed",
            "Stage C 40 CPU models separately approved and passed",
            "Stage D 15 GPU models separately approved and passed",
            "paired 200-well shadow runtime guard PASS",
            "new explicit inference approval",
        ],
    }


# %% [markdown]
# ## 4. Setup and explicit stop

# %%
if EXECUTE_NOTEBOOK:
    CONFIG_PATH = find_config_path()
    CONFIG = read_config(CONFIG_PATH)
    DISABLED = validate_disabled_inference(CONFIG)
    print("Experiment:", EXPERIMENT_NAME)
    print("Route:", get_nested(CONFIG, "experiment.route"))
    print("Parent:", get_nested(CONFIG, "lineage.parent"))
    print("Inference status:", DISABLED)
    raise RuntimeError(
        "exp286 inference is disabled: Stage 0/exp276/Stage B/C/D/runtime guards "
        "and explicit inference approval are not complete"
    )

