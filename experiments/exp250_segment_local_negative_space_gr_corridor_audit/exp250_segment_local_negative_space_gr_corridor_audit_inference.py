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
# # exp250 segment-local negative-space GR corridor audit — inference
#
# This experiment is deliberately train-side diagnostic only. A positive audit
# may start a separate add-only confidence-feature experiment; it never enables
# hidden-test path generation or submission here.

# %% [markdown]
# ## Contents
# 1. Imports and configuration
# 2. Train-side-only inference guard

# %%
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

EXPERIMENT_NAME = "exp250_segment_local_negative_space_gr_corridor_audit"
PACKAGE_DIR = Path.cwd()


def find_config() -> Path:
    candidates = [PACKAGE_DIR / "config.yaml"]
    for parent in PACKAGE_DIR.parents:
        candidates.append(
            parent / "experiments" / EXPERIMENT_NAME / "config.yaml"
        )
        candidates.append(parent / "config.yaml")
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
    raise ValueError("exp250 inference.enabled must remain false")
if config.get("inference", {}).get("create_submission", True):
    raise ValueError("exp250 must never create submission.csv")
raise RuntimeError(
    "exp250 is a train-side corridor audit only. It does not aggregate overlap "
    "views into a rule, change or prune candidates, cut HMM/PF/Beam states, "
    "generate raw-test predictions, or create submission.csv."
)
