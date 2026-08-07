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
# # exp361 exp333 candidate-path add-one novelty audit inference
#
# exp361 is a train-side candidate-novelty audit. This notebook is
# deliberately fail-closed: it validates the disabled inference/submission
# contract and stops without reading raw test, generating a TVT path, or
# writing a submission file.

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

EXPERIMENT_NAME = "exp361_exp333_candidate_path_addone_novelty_audit"


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
    os.environ.get("EXP361_IMPORT_ONLY", "0") != "1" and in_notebook_runtime()
)


def project_root() -> Path:
    start = Path.cwd()
    for candidate in (start, *start.parents):
        if (candidate / "project.yml").exists():
            return candidate
    return start


def find_config_path() -> Path:
    for candidate in (
        Path.cwd() / "config.yaml",
        project_root() / "experiments" / EXPERIMENT_NAME / "config.yaml",
    ):
        if candidate.exists():
            return candidate
    matches = sorted(Path.cwd().rglob(f"{EXPERIMENT_NAME}/config.yaml"))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError("exp361 config.yaml was not found unambiguously")


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
    forbidden = list(get_nested(config, "forbidden_actions") or [])
    checks = {
        "experiment_inference_enabled_false": get_nested(
            config, "experiment.inference_enabled"
        )
        is False,
        "execution_inference_false": get_nested(config, "execution.inference")
        is False,
        "execution_submission_false": get_nested(config, "execution.submission")
        is False,
        "inference_not_authorized": get_nested(
            config, "execution.kaggle_execution_authorization_source"
        )
        != "inference",
        "no_inference_sources": get_nested(
            config, "runtime.kaggle.inference_kernel_sources"
        )
        == [],
        "forbidden_raw_test_inference": "raw_test_inference" in forbidden,
        "forbidden_submission": "submission" in forbidden,
    }
    if not all(checks.values()):
        raise ValueError(f"exp361 disabled inference contract failed: {checks}")
    return checks


def stop_without_inference(config: Mapping[str, Any]) -> None:
    checks = validate_disabled_inference(config)
    print("Experiment:", EXPERIMENT_NAME)
    print("Inference contract checks:", checks)
    raise RuntimeError(
        "exp361 is a train-side exp333 candidate-path audit. Raw-test inference "
        "and submission generation are intentionally disabled."
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
