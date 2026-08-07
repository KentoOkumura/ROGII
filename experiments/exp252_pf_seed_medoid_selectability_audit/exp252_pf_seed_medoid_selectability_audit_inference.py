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
# # exp252 PF seed-medoid selectability audit — inference guard
#
# 本実験はtrain-sideの保存済み候補診断だけを行う。selector、raw-test PF再生成、
# submission作成は別実験のfold-safe設計が承認されるまで禁止する。

# %% [markdown]
# ## Contents
#
# 1. Imports and configuration
# 2. Train-side-only guard

# %% [markdown]
# ## 1. Imports and configuration

# %%
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

EXPERIMENT_NAME = "exp252_pf_seed_medoid_selectability_audit"
PACKAGE_DIR = Path.cwd()


def get_nested(mapping: dict[str, Any], dotted_key: str) -> Any:
    current: Any = mapping
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def find_config_path() -> Path:
    for candidate in (
        PACKAGE_DIR / "config.yaml",
        PACKAGE_DIR / "experiments" / EXPERIMENT_NAME / "config.yaml",
    ):
        if candidate.exists():
            return candidate
    raise FileNotFoundError("exp252 config.yaml was not found")


with find_config_path().open() as fp:
    config = yaml.safe_load(fp) or {}

print("Experiment:", get_nested(config, "experiment.name"))
print("Route:", get_nested(config, "experiment.route"))
print("Inference mode:", get_nested(config, "inference.mode"))
print("Submission enabled:", get_nested(config, "inference.create_submission"))


# %% [markdown]
# ## 2. Train-side-only guard

# %%
assert get_nested(config, "experiment.name") == EXPERIMENT_NAME
assert get_nested(config, "experiment.route") == "pf_beam"
assert get_nested(config, "inference.enabled") is False
assert get_nested(config, "inference.create_submission") is False
assert get_nested(config, "inference.selected_candidate") is None
raise RuntimeError(
    "exp252 inference is disabled: this experiment is a saved-candidate "
    "train-side selectability audit and must not create submission.csv."
)
