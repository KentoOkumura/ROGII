# %% [markdown]
# # exp283 self-GR top-K future-evidence inference boundary
#
# exp283 is a train-side zero-booster diagnostic. This notebook deliberately
# fails closed and cannot create test predictions, decoder output, or submission.

# %% [markdown]
# ## Contents
# 1. Imports and fixed disabled contract
# 2. Configuration validation
# 3. Fail-closed inference boundary

# %% [markdown]
# ## 1. Imports and fixed disabled contract

# %%
from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

EXPERIMENT_NAME = "exp283_self_gr_topk_alternative_mode_branch_future_evidence_readout"


def in_notebook_runtime() -> bool:
    try:
        return get_ipython() is not None  # type: ignore[name-defined]
    except NameError:
        return False


EXECUTE_NOTEBOOK = os.environ.get("EXP283_IMPORT_ONLY", "0") != "1" and in_notebook_runtime()


# %% [markdown]
# ## 2. Configuration validation


# %%
def get_nested(config: Mapping[str, Any], dotted_key: str) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def project_root() -> Path:
    start = Path.cwd()
    for candidate in (start, *start.parents):
        if (candidate / "project.yml").exists():
            return candidate
    return start


def load_config() -> dict[str, Any]:
    candidates = (
        Path.cwd() / "config.yaml",
        project_root() / "experiments" / EXPERIMENT_NAME / "config.yaml",
    )
    for path in candidates:
        if not path.exists():
            continue
        value = yaml.safe_load(path.read_text()) or {}
        if isinstance(value, dict) and get_nested(value, "experiment.name") == EXPERIMENT_NAME:
            return value
    raise FileNotFoundError("exp283 config.yaml was not found")


def assert_inference_disabled(config: Mapping[str, Any]) -> dict[str, Any]:
    contract = {
        "inference_enabled": bool(get_nested(config, "inference.enabled")),
        "create_submission": bool(get_nested(config, "inference.create_submission")),
        "execution_inference": bool(get_nested(config, "execution.inference")),
        "execution_submission": bool(get_nested(config, "execution.submission")),
        "boosters": int(get_nested(config, "execution.total_boosters")),
    }
    if any(contract[key] for key in contract if key != "boosters"):
        raise ValueError("exp283 inference and submission flags must remain disabled")
    if contract["boosters"] != 0:
        raise ValueError("exp283 must remain a zero-booster diagnostic")
    return contract


# %% [markdown]
# ## 3. Fail-closed inference boundary


# %%
def fail_closed() -> None:
    raise RuntimeError(
        "exp283 is a train-side zero-booster branch proposal/evidence readout. "
        "Decoder connection, test inference, corrected predictions, and submission are disabled."
    )


if EXECUTE_NOTEBOOK:
    CONFIG = load_config()
    assert_inference_disabled(CONFIG)
    fail_closed()
