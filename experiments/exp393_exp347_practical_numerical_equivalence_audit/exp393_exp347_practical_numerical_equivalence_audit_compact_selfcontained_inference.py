# %% [markdown]
# # exp393 exp347 practical numerical equivalence audit inference
#
# exp393 Stage A may create one fold-0 model under an explicit user override,
# but Stage B, inference, and submission are not authorized. This notebook is
# deliberately fail-closed and never loads test data or creates predictions.

# %% [markdown]
# ## Contents
# 1. Imports and path helpers
# 2. Configuration lookup
# 3. Fail-closed inference contract
# 4. Disabled inference report

# %% [markdown]
# ## 1. Imports and path helpers

# %%
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml

EXPERIMENT_NAME = "exp393_exp347_practical_numerical_equivalence_audit"


def in_notebook_runtime() -> bool:
    try:
        return get_ipython() is not None  # type: ignore[name-defined]
    except NameError:
        return False


EXECUTE_NOTEBOOK = (
    os.environ.get("EXP393_IMPORT_ONLY", "0") != "1"
    and in_notebook_runtime()
)


def project_root() -> Path:
    start = Path.cwd()
    for candidate in (start, *start.parents):
        if (candidate / "project.yml").exists():
            return candidate
    return start


# %% [markdown]
# ## 2. Configuration lookup

# %%
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
        if value.get("experiment", {}).get("name") == EXPERIMENT_NAME:
            return value
    raise FileNotFoundError(
        f"exp393 config not found in {[str(path) for path in candidates]}"
    )


# %% [markdown]
# ## 3. Fail-closed inference contract

# %%
def validate_disabled_inference(config: dict[str, Any]) -> dict[str, Any]:
    inference = config.get("inference", {})
    execution = config.get("execution", {})
    implementation = config.get("implementation", {})
    if not bool(implementation.get("approved")):
        raise ValueError("exp393 implementation approval is missing")
    if bool(inference.get("enabled")):
        raise ValueError("exp393 inference is not approved")
    if bool(inference.get("create_submission")):
        raise ValueError("exp393 must not create a submission")
    if bool(execution.get("inference_approved")):
        raise ValueError("execution.inference_approved must remain false")
    if bool(execution.get("submission_approved")):
        raise ValueError("execution.submission_approved must remain false")
    if bool(execution.get("run_inference")):
        raise ValueError("execution.run_inference must remain false")
    if bool(execution.get("create_submission")):
        raise ValueError("execution.create_submission must remain false")
    trained_folds = int(execution.get("current_trained_fold_count", -1))
    if trained_folds not in {0, 1}:
        raise ValueError("exp393 may contain at most the authorized fold-0 model")
    if inference.get("prerequisite") != (
        "stage_a_then_stage_b_promotion_and_separate_approval"
    ):
        raise ValueError("exp393 inference prerequisite changed")
    return {
        "experiment": EXPERIMENT_NAME,
        "route": config.get("experiment", {}).get("route"),
        "inference_enabled": False,
        "create_submission": False,
        "persistent_model_count": trained_folds,
        "trained_fold_count": trained_folds,
        "stage0_run_approved": False,
        "stage_a_authorized": bool(execution.get("stage_a_gpu_approved")),
        "reason": inference.get("reason"),
    }


# %% [markdown]
# ## 4. Disabled inference report

# %%
if EXECUTE_NOTEBOOK:
    CONFIG = load_config()
    REPORT = validate_disabled_inference(CONFIG)
    print(json.dumps(REPORT, indent=2, sort_keys=True))
