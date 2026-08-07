# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # exp377 formation-relative K16 slope identifiability inference
#
# exp377 is a train-side identifiability readout. It deliberately has no
# current-test generation, inference candidate, or submission path.

# %% [markdown]
# ## Contents
#
# 1. Imports and runtime helpers
# 2. Configuration contract
# 3. Exp377 result prerequisite
# 4. Inference prohibition
# 5. Generated artifacts

# %% [markdown]
# ## 1. Imports and runtime helpers

# %%
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

EXPERIMENT_NAME = "exp377_formation_relative_k16_slope_identifiability_readout"


def project_root() -> Path:
    for candidate in (Path.cwd(), *Path.cwd().parents):
        if (candidate / "project.yml").exists():
            return candidate
    return Path.cwd()


def find_config_path() -> Path:
    candidates = [
        Path.cwd() / "config.yaml",
        Path("/kaggle/working/config.yaml"),
        project_root() / "experiments" / EXPERIMENT_NAME / "config.yaml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("exp377 config.yaml was not found")


def read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text()) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def get_nested(
    mapping: Mapping[str, Any],
    dotted_key: str,
    default: Any = None,
) -> Any:
    current: Any = mapping
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


# %% [markdown]
# ## 2. Configuration contract

# %%
CONFIG_PATH = find_config_path()
CONFIG = read_yaml(CONFIG_PATH)
print("Experiment:", EXPERIMENT_NAME)
print("Route:", get_nested(CONFIG, "experiment.route"))
print("Status:", get_nested(CONFIG, "experiment.status"))
print("Inference enabled:", get_nested(CONFIG, "execution.inference_enabled"))
print("Submission enabled:", get_nested(CONFIG, "execution.submission_enabled"))

if get_nested(CONFIG, "experiment.route") != "pf_beam":
    raise ValueError("exp377 route must remain pf_beam")
if bool(get_nested(CONFIG, "execution.inference_enabled")):
    raise ValueError("exp377 inference must remain disabled")
if bool(get_nested(CONFIG, "execution.submission_enabled")):
    raise ValueError("exp377 submission must remain disabled")


# %% [markdown]
# ## 3. Exp377 result prerequisite

# %%
summary_candidates = [
    Path("/kaggle/working/artifacts")
    / f"{EXPERIMENT_NAME}_summary.json",
    project_root()
    / "experiments"
    / EXPERIMENT_NAME
    / "artifacts"
    / f"{EXPERIMENT_NAME}_summary.json",
]
summary_path = next(
    (candidate for candidate in summary_candidates if candidate.exists()),
    None,
)
if summary_path is None:
    print("No completed exp377 summary is available.")
else:
    summary = json.loads(summary_path.read_text())
    print("Train-side decision:", summary.get("decision"))
    print("Stage 0:", summary.get("stage0", {}).get("passed"))
    stage1 = summary.get("stage1")
    print("Stage 1:", None if stage1 is None else stage1.get("passed"))


# %% [markdown]
# ## 4. Inference prohibition

# %%
raise RuntimeError(
    "exp377 is diagnostic-only. It must not generate current-test rates, "
    "submission.csv, or any inference candidate. A passing exp377 may only "
    "unblock a separately approved exp378 implementation."
)


# %% [markdown]
# ## 5. Generated artifacts
#
# None. This fail-closed notebook writes no prediction or submission file.
