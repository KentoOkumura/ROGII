# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # exp396 fold-safe exp111 score 27 add-only on exp287 — inference
#
# Stage A実装時点ではcurrent-test score、Stage B TVT model、submissionのいずれも未承認である。
# この候補Notebookはfail-closed contractだけを実装し、予測fileを生成しない。

# %% [markdown]
# ## Contents
#
# 1. Imports and configuration
# 2. Inference approval contract
# 3. Current-test design boundary
# 4. Fail-closed orchestration

# %% [markdown]
# ## 1. Imports and configuration

# %%
from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from IPython.display import display

EXPERIMENT_NAME = "exp396_fold_safe_exp111_score_27_addonly_on_exp287"
PACKAGE_DIR = Path.cwd()
IMPORT_ONLY = os.environ.get("EXP396_INFERENCE_IMPORT_ONLY", "0") == "1"


def read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def find_config_path() -> Path:
    candidates = [
        PACKAGE_DIR / "config.yaml",
        PACKAGE_DIR / "experiments" / EXPERIMENT_NAME / "config.yaml",
    ]
    candidates.extend(PACKAGE_DIR.rglob(f"{EXPERIMENT_NAME}/config.yaml"))
    matches = sorted({path.resolve() for path in candidates if path.exists()})
    if len(matches) != 1:
        raise FileNotFoundError(f"exp396 config resolution is ambiguous: {matches}")
    return matches[0]


# %% [markdown]
# ## 2. Inference approval contract
#
# inferenceに進めるのはStage AとStage Bの全gateがPASSし、outer fold別4 scorer ensembleと
# 保存済み15 TVT modelsを使うcurrent-test実装が別承認された後だけである。

# %%
def validate_inference_is_closed(config: Mapping[str, Any]) -> dict[str, Any]:
    if config["experiment"]["name"] != EXPERIMENT_NAME:
        raise ValueError("experiment name contract changed")
    if config["experiment"]["route"] != "ml_model":
        raise ValueError("exp396 route must remain ml_model")
    execution = dict(config["execution"])
    forbidden_enabled = {
        "run_inference": bool(execution["run_inference"]),
        "create_submission": bool(execution["create_submission"]),
        "submit_to_kaggle": bool(execution["submit_to_kaggle"]),
        "inference_approved": bool(execution["inference_approved"]),
        "submission_approved": bool(execution["submission_approved"]),
    }
    if any(forbidden_enabled.values()):
        raise RuntimeError(
            "exp396 inference/submission is not implemented or authorized: "
            f"{forbidden_enabled}"
        )
    return {
        "status": "inference_not_implemented_not_approved",
        "stage_a_current_test_scorer_models": 0,
        "stage_b_saved_tvt_models": 0,
        "boosters_trained": 0,
        "prediction_generated": False,
        "submission_generated": False,
    }


# %% [markdown]
# ## 3. Current-test design boundary
#
# 将来の別承認時もfull-train scorer refitは行わず、downstream outer foldごとの
# 保存済み4 inner model pair平均をcurrent-testへ適用する。現時点ではmodel/cacheを読まない。

# %%
def current_test_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    current_test = dict(config["scorer"]["current_test"])
    if bool(current_test["enabled"]):
        raise RuntimeError("current-test score generation has not been approved")
    if bool(current_test["full_train_refit"]):
        raise ValueError("full-train scorer refit is forbidden")
    if not bool(current_test["use_outer_fold_specific_four_model_ensemble"]):
        raise ValueError("future current-test contract must retain outer-specific four-model mean")
    return {
        "enabled": False,
        "full_train_refit": False,
        "outer_fold_specific_model_pairs": 0,
        "prediction_generated": False,
    }


# %% [markdown]
# ## 4. Fail-closed orchestration

# %%
if not IMPORT_ONLY:
    CONFIG = read_yaml(find_config_path())
    INFERENCE_STATUS = validate_inference_is_closed(CONFIG)
    CURRENT_TEST_STATUS = current_test_contract(CONFIG)
    display(INFERENCE_STATUS)
    display(CURRENT_TEST_STATUS)
    print("No prediction or submission file was generated.")
