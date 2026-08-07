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
# # exp254 Numba all-seed PF speed reproduction inference
#
# exp254はtrain-sideのruntime / parity / determinism基盤監査であり、raw-test
# inference、candidate選択、submission生成を禁止する。

# %% [markdown]
# ## Contents
#
# 1. Imports
# 2. Configuration contract
# 3. Disabled inference guard

# %% [markdown]
# ## 1. Imports

# %%
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

# %% [markdown]
# ## 2. Configuration contract

# %%
EXPERIMENT_NAME = "exp254_numba_allseed_pf_speed_reproduction"
PACKAGE_DIR = Path.cwd()


def find_config_path() -> Path:
    candidates = [
        PACKAGE_DIR / "config.yaml",
        PACKAGE_DIR / "experiments" / EXPERIMENT_NAME / "config.yaml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"config.yaml not found; checked={candidates}")


def get_nested(mapping: dict[str, Any], dotted_key: str) -> Any:
    current: Any = mapping
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


with find_config_path().open() as fp:
    config = yaml.safe_load(fp) or {}
if not isinstance(config, dict):
    raise ValueError("config.yaml must contain a mapping")

print(
    {
        "experiment": get_nested(config, "experiment.name"),
        "route": get_nested(config, "experiment.route"),
        "parent": get_nested(config, "lineage.parent"),
        "inference_enabled": get_nested(config, "inference.enabled"),
        "create_submission": get_nested(config, "inference.create_submission"),
        "runtime_only": True,
    }
)

# %% [markdown]
# ## 3. Disabled inference guard

# %%
if bool(get_nested(config, "inference.enabled")):
    raise RuntimeError("exp254 inference must remain disabled")
if bool(get_nested(config, "inference.create_submission")):
    raise RuntimeError("exp254 must never create submission.csv")
if os.environ.get("EXP254_ALLOW_INFERENCE", "0") == "1":
    raise RuntimeError("exp254 has no inference implementation to enable")

raise RuntimeError(
    "Inference is intentionally disabled: exp254 is a PF runtime/parity foundation "
    "audit only and does not generate raw-test predictions or submission.csv."
)
