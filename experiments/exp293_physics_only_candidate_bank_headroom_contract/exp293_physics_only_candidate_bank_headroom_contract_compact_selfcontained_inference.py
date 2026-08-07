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
# # exp293 physics-only candidate-bank headroom inference
#
# exp293 is a train-side oracle-headroom audit. This notebook is deliberately
# fail-closed: it validates the disabled inference/submission contract and then
# stops without reading raw test, creating a TVT prediction, or writing a
# submission file.

# %% [markdown]
# ## Contents
# 1. Imports
# 2. Notebook-safe configuration helpers
# 3. Disabled inference contract
# 4. Execution

# %%
from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

EXPERIMENT_NAME = "exp293_physics_only_candidate_bank_headroom_contract"


# %% [markdown]
# ## 2. Notebook-safe configuration helpers

# %%
def in_notebook_runtime() -> bool:
    try:
        shell = get_ipython()  # type: ignore[name-defined]
    except NameError:
        return False
    return shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell"


EXECUTE_NOTEBOOK = (
    os.environ.get("EXP293_IMPORT_ONLY", "0") != "1" and in_notebook_runtime()
)


def project_root() -> Path:
    start = Path.cwd()
    for candidate in (start, *start.parents):
        if (candidate / "project.yml").exists():
            return candidate
    return start


def find_config_path() -> Path:
    direct = Path.cwd() / "config.yaml"
    if direct.exists():
        return direct
    nested = project_root() / "experiments" / EXPERIMENT_NAME / "config.yaml"
    if nested.exists():
        return nested
    matches = sorted(Path.cwd().rglob(f"{EXPERIMENT_NAME}/config.yaml"))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError("exp293 config.yaml was not found unambiguously")


def read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text()) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def get_nested(mapping: Mapping[str, Any], dotted_key: str) -> Any:
    current: Any = mapping
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


# %% [markdown]
# ## 3. Disabled inference contract

# %%
def validate_disabled_inference(config: Mapping[str, Any]) -> dict[str, Any]:
    checks = {
        "experiment_inference_enabled_false": get_nested(
            config, "experiment.inference_enabled"
        )
        is False,
        "execution_inference_false": get_nested(config, "execution.inference")
        is False,
        "execution_submission_false": get_nested(config, "execution.submission")
        is False,
        "kaggle_inference_push_unapproved": get_nested(
            config, "execution.kaggle_inference_push_approved"
        )
        is False,
        "canonical_inference_unadopted": get_nested(
            config, "execution.canonical_inference_notebook_adopted"
        )
        is False,
        "no_inference_sources": get_nested(
            config, "runtime.kaggle.inference_kernel_sources"
        )
        == [],
        "forbidden_raw_test_inference": "raw_test_inference"
        in get_nested(config, "audit.forbidden_actions"),
        "forbidden_submission": "submission"
        in get_nested(config, "audit.forbidden_actions"),
    }
    if not all(checks.values()):
        raise ValueError(f"exp293 disabled inference contract failed: {checks}")
    return checks


def stop_without_inference(config: Mapping[str, Any]) -> None:
    checks = validate_disabled_inference(config)
    print("Experiment:", EXPERIMENT_NAME)
    print("Inference contract checks:", checks)
    raise RuntimeError(
        "exp293 is a train-side headroom audit. Raw-test inference and "
        "submission generation are intentionally disabled."
    )


# %% [markdown]
# ## 4. Execution

# %%
if EXECUTE_NOTEBOOK:
    CONFIG_PATH = find_config_path()
    CONFIG = read_yaml(CONFIG_PATH)
    print("Config:", CONFIG_PATH)
    print("Config SHA256:", sha256_file(CONFIG_PATH))
    stop_without_inference(CONFIG)
