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
# # exp247 missing GR masking — inference guard
#
# exp247 is a train-side one-change ablation. A positive OOF result does not
# authorize current-test generation because exp221 itself showed weak CV-to-LB
# transfer. This notebook therefore documents and enforces the no-inference
# contract.

# %% [markdown]
# ## Contents
# 1. Imports and configuration
# 2. Train-side-only inference guard

# %%
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

EXPERIMENT_NAME = "exp247_missing_gr_masking"
PACKAGE_DIR = Path.cwd()


def find_config() -> Path:
    candidates = [
        PACKAGE_DIR / "config.yaml",
        *[
            parent / "experiments" / EXPERIMENT_NAME / "config.yaml"
            for parent in PACKAGE_DIR.parents
        ],
    ]
    for path in candidates:
        if not path.exists():
            continue
        value: dict[str, Any] = yaml.safe_load(path.read_text()) or {}
        if value.get("experiment", {}).get("name") == EXPERIMENT_NAME:
            return path
    raise FileNotFoundError(f"Could not resolve config.yaml for {EXPERIMENT_NAME}")


config: dict[str, Any] = yaml.safe_load(find_config().read_text()) or {}
print("Experiment:", config.get("experiment", {}).get("name"))
print("Route:", config.get("experiment", {}).get("route"))
print("Inference contract:", config.get("inference", {}))

# %% [markdown]
# ## 2. Train-side-only inference guard

# %%
if config.get("inference", {}).get("enabled", True):
    raise ValueError("exp247 inference.enabled must remain false")
raise RuntimeError(
    "exp247 is a train-side missing-GR exact-HMM ablation only. It does not "
    "generate current-test predictions or submission.csv. Review the full OOF "
    "missing-run, hidden-like, long-tail, and worst-well readout first."
)
