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
# # exp298 inference is intentionally disabled
#
# exp298 is a post-freeze train-side diagnostic. It creates no deployable
# prediction, model, selector, correction, or submission. This fail-closed
# notebook exists only to make the disabled inference contract explicit.

# %% [markdown]
# ## Contents
# 1. Imports and runtime helpers
# 2. Disabled inference contract
# 3. Contract preview and fail-closed stop

# %%
from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

EXPERIMENT_NAME = "exp298_exp226_blockwise_offset_slope_quotient_local_shape_audit"


# %% [markdown]
# ## 1. Imports and runtime helpers

# %%
def in_notebook_runtime() -> bool:
    try:
        shell = get_ipython()  # type: ignore[name-defined]
    except NameError:
        return False
    return shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell"


EXECUTE_NOTEBOOK = (
    os.environ.get("EXP298_IMPORT_ONLY", "0") != "1" and in_notebook_runtime()
)


def experiment_dir() -> Path:
    start = Path.cwd()
    for candidate in (start, *start.parents):
        nested = candidate / "experiments" / EXPERIMENT_NAME
        if nested.exists():
            return nested
    return Path.cwd()


def find_config_path() -> Path:
    candidates = [Path.cwd() / "config.yaml", experiment_dir() / "config.yaml"]
    matches = [path for path in candidates if path.exists()]
    if matches:
        return matches[0]
    recursive = sorted(Path.cwd().rglob(f"{EXPERIMENT_NAME}/config.yaml"))
    if len(recursive) == 1:
        return recursive[0]
    raise FileNotFoundError("exp298 config.yaml was not found unambiguously")


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


# %% [markdown]
# ## 2. Disabled inference contract

# %%
def validate_disabled_inference(config: Mapping[str, Any]) -> dict[str, bool]:
    expected_artifacts = set(
        get_nested(config, "audit.expected_artifacts_if_implemented") or []
    )
    checks = {
        "route_is_pf_beam": get_nested(config, "experiment.route") == "pf_beam",
        "experiment_inference_disabled": not bool(
            get_nested(config, "experiment.inference_enabled")
        ),
        "execution_inference_disabled": not bool(
            get_nested(config, "execution.inference")
        ),
        "execution_submission_disabled": not bool(
            get_nested(config, "execution.submission")
        ),
        "canonical_inference_notebook_unadopted": not bool(
            get_nested(config, "execution.canonical_inference_notebook_adopted")
        ),
        "inference_kernel_sources_empty": not bool(
            get_nested(config, "runtime.kaggle.inference_kernel_sources")
        ),
        "submission_not_expected": "submission.csv" not in expected_artifacts,
        "zero_models": int(get_nested(config, "execution.total_boosters")) == 0,
        "zero_pf_beam_reruns": int(
            get_nested(config, "execution.hmm_pf_well_runs")
        )
        == 0,
    }
    if not all(checks.values()):
        raise ValueError(f"disabled inference contract mismatch: {checks}")
    return checks


def stop_without_inference(config: Mapping[str, Any]) -> None:
    checks = validate_disabled_inference(config)
    print(json.dumps(checks, indent=2, sort_keys=True))
    raise RuntimeError(
        "exp298 inference is intentionally disabled: the audit produces only "
        "diagnostic quotient metrics and no deployable TVT prediction."
    )


# %% [markdown]
# ## 3. Contract preview and fail-closed stop

# %%
if EXECUTE_NOTEBOOK:
    CONFIG_PATH = find_config_path()
    CONFIG = read_yaml(CONFIG_PATH)
    print("Experiment:", EXPERIMENT_NAME)
    print("Config:", CONFIG_PATH)
    stop_without_inference(CONFIG)
