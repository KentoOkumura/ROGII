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
# # exp336 exp287 formation-tail attribution readout — inference
#
# exp336 は保存済み OOF に対する診断 readout であり、hidden-test prediction、
# submission、モデル推論を持たない。この notebook は誤って inference package として
# 実行された場合に必ず停止する fail-closed entrypoint である。

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

EXPERIMENT_NAME = "exp336_exp287_formation_tail_attribution_readout"
PACKAGE_DIR = Path.cwd()
IMPORT_ONLY = os.environ.get("EXP336_IMPORT_ONLY", "0") == "1"


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
        raise FileNotFoundError(f"exp336 config resolution is ambiguous: {matches}")
    return matches[0].resolve()


# %% [markdown]
# ## 2. Zero-inference contract audit


# %%
def validate_zero_inference_contract(config: dict[str, Any]) -> dict[str, Any]:
    if config.get("experiment", {}).get("name") != EXPERIMENT_NAME:
        raise ValueError("experiment name contract changed")
    if config.get("experiment", {}).get("route") != "ml_model":
        raise ValueError("exp336 route must remain ml_model")
    model = dict(config.get("model") or {})
    counts = {
        "active_variants": int(model.get("active_variants", -1)),
        "lightgbm_configs": int(model.get("lightgbm_config_count", -1)),
        "trained_folds": int(model.get("trained_fold_count", -1)),
        "boosters": int(model.get("booster_count", -1)),
    }
    if counts != {
        "active_variants": 0,
        "lightgbm_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
    }:
        raise ValueError(f"zero-model contract changed: {counts}")
    execution = dict(config.get("execution") or {})
    forbidden = {
        name: bool(execution.get(name, False))
        for name in [
            "run_model_training",
            "run_inference",
            "create_submission",
            "submit_to_kaggle",
        ]
    }
    enabled = [name for name, value in forbidden.items() if value]
    if enabled:
        raise ValueError(f"exp336 forbidden inference flags are enabled: {enabled}")
    outputs = set(config.get("audit", {}).get("expected_outputs") or [])
    forbidden_outputs = sorted(
        name
        for name in outputs
        if "submission" in str(name).lower() or "prediction" in str(name).lower()
    )
    if forbidden_outputs:
        raise ValueError(
            f"exp336 output contract contains prediction/submission: {forbidden_outputs}"
        )
    return {
        **counts,
        "run_inference": False,
        "create_submission": False,
        "submit_to_kaggle": False,
        "expected_diagnostic_outputs": len(outputs),
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
            "status": "intentional_fail_closed_inference",
        },
        sort_keys=True,
        indent=2,
    )
)


# %% [markdown]
# ## 3. Intentional fail-closed stop
#
# この停止は実装上のエラーではない。exp336 から submission が生成されないことを
# notebook 実行時にも保証するための契約である。

# %%
if not IMPORT_ONLY:
    raise RuntimeError(
        "exp336 is an OOF diagnostic readout. Inference, prediction, and submission "
        "generation are intentionally forbidden."
    )
