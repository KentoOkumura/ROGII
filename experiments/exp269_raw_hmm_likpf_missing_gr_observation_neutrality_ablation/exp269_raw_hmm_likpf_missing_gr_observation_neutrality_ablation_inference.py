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
# # exp269 raw HMM missing-GR neutrality — inference/PF guard
#
# exp269 Stage 1 is a train-side one-change ablation on the exp209 raw exact HMM.
# A positive paired result records likelihood-PF eligibility only; it does not
# authorize PF Stage 2 or current-test generation. This notebook documents and
# enforces that fail-closed contract.

# %% [markdown]
# ## Contents
# 1. Imports and configuration
# 2. Train-side-only inference guard

# %%
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

EXPERIMENT_NAME = "exp269_raw_hmm_likpf_missing_gr_observation_neutrality_ablation"
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
print("PF Stage contract:", config.get("pf_stage", {}))
print("Inference contract:", config.get("inference", {}))

# %% [markdown]
# ## 2. Train-side-only inference guard

# %%
if config.get("inference", {}).get("enabled", True):
    raise ValueError("exp269 inference.enabled must remain false")
if config.get("pf_stage", {}).get("enabled", True):
    raise ValueError("exp269 pf_stage.enabled must remain false until separate approval")
raise RuntimeError(
    "exp269 currently implements Stage 1 only. It does not run likelihood-PF, "
    "generate current-test predictions, or create submission.csv. Review the "
    "pre-registered Stage-1 guard first, then request a separately approved PF run."
)
