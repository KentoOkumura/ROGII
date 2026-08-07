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
# # exp249 segment-local negative-space GR corridor audit — inference
#
# exp249 is deliberately train-side diagnostic only. Segment-local signals are
# not allowed to alter hidden-test paths, prune candidates, or create a
# submission. A positive Stage 1 result would start a separate add-only feature
# experiment rather than enabling inference here.

# %% [markdown]
# ## Contents
# 1. Imports and configuration
# 2. Diagnostic-only inference guard

# %%
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

EXPERIMENT_NAME = "exp249_segment_local_negative_space_gr_corridor_audit"
PACKAGE_DIR = Path.cwd()


def find_config() -> Path:
    candidates = [
        PACKAGE_DIR / "config.yaml",
        *PACKAGE_DIR.parents,
    ]
    for candidate in candidates:
        path = candidate if candidate.name == "config.yaml" else candidate / "config.yaml"
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
# ## 2. Diagnostic-only inference guard

# %%
if config.get("inference", {}).get("enabled", True):
    raise ValueError("exp249 inference.enabled must remain false")
raise RuntimeError(
    "exp249 is a train-side segment-local negative-space diagnostic only. It does "
    "not aggregate overlap views, prune or average candidates, apply HMM/PF/Beam "
    "edge cuts, generate hidden-test predictions, or create submission.csv."
)
