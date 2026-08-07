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
# # exp334 equal-well loss weighting on exp287 — inference
#
# exp334 は train-side promotion gate 未評価であり、inference / submission は未承認である。
# この notebook 候補は誤って sample submission や予測を生成しない fail-closed entrypoint とする。

# %% [markdown]
# ## Contents
#
# 1. Imports and configuration helpers
# 2. Authorization contract
# 3. Train-result gate contract
# 4. Inference status

# %% [markdown]
# ## 1. Imports and configuration helpers

# %%
from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from IPython.display import display

EXPERIMENT_NAME = "exp334_equal_well_loss_weighting_on_exp287"
PACKAGE_DIR = Path.cwd()
IMPORT_ONLY = os.environ.get("EXP334_IMPORT_ONLY", "0") == "1"


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
        raise FileNotFoundError(f"exp334 config resolution is ambiguous: {matches}")
    return matches[0]


# %% [markdown]
# ## 2. Authorization contract
#
# train guard PASS、保存model/feature schema SHA固定、別途inference実装承認が揃うまで常に停止する。

# %%
def validate_inference_is_closed(config: Mapping[str, Any]) -> dict[str, Any]:
    execution = dict(config["execution"])
    if str(config["experiment"]["name"]) != EXPERIMENT_NAME:
        raise ValueError("experiment name contract changed")
    if str(config["experiment"]["route"]) != "ml_model":
        raise ValueError("exp334 route must remain ml_model")
    prohibited_flags = {
        "run_inference": bool(execution["run_inference"]),
        "create_submission": bool(execution["create_submission"]),
        "submit_to_kaggle": bool(execution["submit_to_kaggle"]),
    }
    if any(prohibited_flags.values()):
        raise RuntimeError(
            "exp334 inference is not implemented or authorized; reset all inference/submission "
            f"flags to false: {prohibited_flags}"
        )
    return {
        "status": "fail_closed_waiting_train_guard_and_separate_approval",
        "run_inference": False,
        "create_submission": False,
        "submit_to_kaggle": False,
        "prediction_generated": False,
        "submission_generated": False,
    }


# %% [markdown]
# ## 3. Train-result gate contract
#
# 将来のinference実装では、train `metrics.json` の全promotion check PASS、15 model manifest、
# 421 feature schema、weight manifest、OOF prediction SHAを固定してから current-test featureを
# exp287と同じ方法で再生成する。今回の実装範囲には含めない。

# %%
INFERENCE_PREREQUISITES = {
    "train_guard_passed": False,
    "saved_model_count": 15,
    "feature_count": 421,
    "model_manifest_sha256_pinned": False,
    "oof_prediction_sha256_pinned": False,
    "separate_user_approval_required": True,
    "submission_requires_additional_approval": True,
}


# %% [markdown]
# ## 4. Inference status

# %%
if not IMPORT_ONLY:
    CONFIG = read_yaml(find_config_path())
    CLOSED_STATUS = validate_inference_is_closed(CONFIG)
    display(
        {
            "experiment": EXPERIMENT_NAME,
            "route": CONFIG["experiment"]["route"],
            "parent": CONFIG["lineage"]["parent"],
            "authorization": CLOSED_STATUS,
            "prerequisites": INFERENCE_PREREQUISITES,
        }
    )
    raise RuntimeError(
        "exp334 inference is fail-closed until train promotion PASS and separate user approval"
    )
