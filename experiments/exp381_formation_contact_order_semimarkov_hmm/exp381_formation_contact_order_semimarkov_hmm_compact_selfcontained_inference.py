# ---
# jupyter:
#   jupytext:
#     formats: py:percent
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
# # exp381 formation contact order semi-Markov HMM: inference guard
#
# exp381 currently contains only a Stage 0 diagnostic. This notebook validates
# that no Stage 1 HMM, inference path, or submission has been authorized and
# then fails closed.

# %% [markdown]
# ## Contents
# 1. Imports and paths
# 2. Configuration and Stage 0 evidence
# 3. Inference authorization guard
# 4. Fail-closed outcome

# %% [markdown]
# ## 1. Imports and paths

# %%
from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

EXPERIMENT_NAME = "exp381_formation_contact_order_semimarkov_hmm"
EXECUTE_NOTEBOOK = os.environ.get("EXP381_IMPORT_ONLY", "0") != "1"


def project_root() -> Path:
    start = Path.cwd().resolve()
    for candidate in (start, *start.parents):
        if (candidate / "project.yml").exists() and (candidate / "AGENTS.md").exists():
            return candidate
    return start


def find_config_path() -> Path:
    root = project_root()
    candidates = [
        root / "experiments" / EXPERIMENT_NAME / "config.yaml",
        Path("/kaggle/working/config.yaml"),
        root / "config.yaml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"config.yaml not found in {candidates}")


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open() as stream:
        value = yaml.safe_load(stream) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def get_nested(
    mapping: Mapping[str, Any],
    dotted_key: str,
    default: Any = None,
) -> Any:
    current: Any = mapping
    for key in dotted_key.split("."):
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]
    return current


def runtime_artifacts_dir() -> Path:
    if Path("/kaggle/working").exists():
        return Path("/kaggle/working/artifacts")
    return project_root() / "experiments" / EXPERIMENT_NAME / "artifacts"


# %% [markdown]
# ## 2. Configuration and Stage 0 evidence

# %%
def load_stage0_summary() -> dict[str, Any] | None:
    path = runtime_artifacts_dir() / f"{EXPERIMENT_NAME}_summary.json"
    if not path.exists():
        return None
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


# %% [markdown]
# ## 3. Inference authorization guard

# %%
def validate_inference_is_disabled(config: Mapping[str, Any]) -> None:
    if get_nested(config, "experiment.route") != "pf_beam":
        raise ValueError("exp381 route must remain pf_beam")
    if int(get_nested(config, "runtime.hmm_runs", -1)) != 0:
        raise ValueError("exp381 Stage 0 config must contain zero HMM runs")
    if bool(get_nested(config, "execution.stage1_implementation_authorized")):
        raise ValueError(
            "Stage 1 authorization changed without an implemented HMM contract"
        )
    if bool(get_nested(config, "execution.inference_enabled")):
        raise ValueError("inference cannot be enabled before Stage 1 exists")
    if bool(get_nested(config, "execution.submission_enabled")):
        raise ValueError("submission cannot be enabled before Stage 1 exists")


def fail_closed(config: Mapping[str, Any]) -> None:
    validate_inference_is_disabled(config)
    summary = load_stage0_summary()
    if summary is None:
        evidence = "Stage 0 has not run."
    else:
        evidence = (
            "Stage 0 passed="
            f"{summary.get('stage0', {}).get('passed')}; "
            f"decision={summary.get('decision')}."
        )
    raise RuntimeError(
        "exp381 has no inference candidate. "
        f"{evidence} A Stage 0 PASS and separate approval are required before "
        "the seven-state semi-Markov HMM may be implemented; inference and "
        "submission remain disabled."
    )


# %% [markdown]
# ## 4. Fail-closed outcome

# %%
if EXECUTE_NOTEBOOK:
    CONFIG_PATH = find_config_path()
    CONFIG = read_yaml(CONFIG_PATH)
    print("Experiment:", EXPERIMENT_NAME)
    print("Route:", get_nested(CONFIG, "experiment.route"))
    print("Stage 1 implemented:", False)
    print("Inference enabled:", get_nested(CONFIG, "execution.inference_enabled"))
    print("Submission enabled:", get_nested(CONFIG, "execution.submission_enabled"))
    fail_closed(CONFIG)
