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
# # exp349 exp287 U-boundary continuity fade — inference guard
#
# exp349 の raw-test inference は Stage 0 の全 gate PASS と別承認後にだけ設計する契約
# だったが、Stage 0 は pooled gain gate を FAIL した。この候補 notebook は hidden-test
# prediction と submission を永久に拒否する fail-closed entrypoint である。

# %% [markdown]
# ## Contents
#
# 1. Imports and configuration resolution
# 2. Zero-inference contract audit
# 3. Intentional fail-closed stop

# %% [markdown]
# ## 1. Imports and configuration resolution

# %%
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml

EXPERIMENT_NAME = "exp349_exp287_u_boundary_continuity_fade"
PACKAGE_DIR = Path.cwd()
IMPORT_ONLY = os.environ.get("EXP349_IMPORT_ONLY", "0") == "1"


def read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def find_config_path() -> Path:
    direct = [
        PACKAGE_DIR / "config.yaml",
        PACKAGE_DIR / "experiments" / EXPERIMENT_NAME / "config.yaml",
    ]
    for candidate in direct:
        if candidate.is_file():
            return candidate.resolve()
    matches = sorted(PACKAGE_DIR.rglob(f"{EXPERIMENT_NAME}/config.yaml"))
    if len(matches) != 1:
        raise FileNotFoundError(f"exp349 config resolution is ambiguous: {matches}")
    return matches[0].resolve()


# %% [markdown]
# ## 2. Zero-inference contract audit

# %%
def validate_zero_inference_contract(config: dict[str, Any]) -> dict[str, Any]:
    if config.get("experiment", {}).get("name") != EXPERIMENT_NAME:
        raise ValueError("experiment name contract changed")
    if config.get("experiment", {}).get("route") != "ml_model":
        raise ValueError("exp349 route must remain ml_model")
    execution = dict(config.get("execution") or {})
    counts = {
        "postprocess_variants": int(execution.get("active_postprocess_variants", -1)),
        "trained_folds": int(execution.get("trained_folds", -1)),
        "model_configs": int(execution.get("model_configs", -1)),
        "trained_models": int(execution.get("trained_models", -1)),
        "boosters": int(execution.get("lightgbm_boosters", -1)),
        "pf_well_runs": int(execution.get("pf_well_runs", -1)),
        "beam_well_runs": int(execution.get("beam_well_runs", -1)),
        "hmm_well_runs": int(execution.get("hmm_well_runs", -1)),
    }
    if counts != {
        "postprocess_variants": 1,
        "trained_folds": 0,
        "model_configs": 0,
        "trained_models": 0,
        "boosters": 0,
        "pf_well_runs": 0,
        "beam_well_runs": 0,
        "hmm_well_runs": 0,
    }:
        raise ValueError(f"zero-training contract changed: {counts}")
    forbidden_flags = [
        "run_model_training",
        "run_inference",
        "create_submission",
        "submit_to_kaggle",
    ]
    enabled = [name for name in forbidden_flags if bool(execution.get(name, False))]
    if enabled:
        raise ValueError(f"exp349 forbidden inference flags are enabled: {enabled}")
    approvals = dict(config.get("implementation", {}).get("approvals") or {})
    if bool(approvals.get("inference")) or bool(approvals.get("submission")):
        raise ValueError("inference/submission approval must remain false before Stage 0 PASS")
    return {
        **counts,
        "run_inference": False,
        "create_submission": False,
        "submit_to_kaggle": False,
        "status": "intentional_fail_closed_after_stage0_scientific_gate_failure",
    }


CONFIG_PATH = find_config_path()
CONFIG = read_yaml(CONFIG_PATH)
ZERO_INFERENCE_CONTRACT = validate_zero_inference_contract(CONFIG)
print(
    json.dumps(
        {
            "experiment": EXPERIMENT_NAME,
            "config_path": str(CONFIG_PATH),
            "contract": ZERO_INFERENCE_CONTRACT,
        },
        sort_keys=True,
        indent=2,
    )
)


# %% [markdown]
# ## 3. Intentional fail-closed stop
#
# この停止は実装上のエラーではない。Stage 0 FAIL後のexp349からraw-test prediction
# またはsubmissionが誤生成されないことをnotebook実行時にも保証する。

# %%
if not IMPORT_ONLY:
    raise RuntimeError(
        "exp349 raw-test inference is blocked because Stage 0 failed its pooled "
        "scientific gain gate; no rescue, inference, or submission is authorized."
    )
