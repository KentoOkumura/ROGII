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
# # exp297 inference (fail closed)
#
# exp297 is a train-side Stage-2 diagnostic. It is forbidden to generate raw-test
# features, TVT predictions, or a submission. This notebook exists only to make
# that execution boundary explicit and machine-checkable.

# %%
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import yaml

EXPERIMENT_NAME = "exp297_prefix_calibrated_latent_registration_gr_evidence"


def in_notebook_runtime() -> bool:
    try:
        shell = get_ipython()  # type: ignore[name-defined]
    except NameError:
        return False
    return shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell"


EXECUTE_NOTEBOOK = os.environ.get("EXP297_IMPORT_ONLY", "0") != "1" and in_notebook_runtime()


def find_config_path() -> Path:
    start = Path.cwd()
    candidates = [start / "config.yaml"]
    candidates.extend(
        parent / "experiments" / EXPERIMENT_NAME / "config.yaml"
        for parent in (start, *start.parents)
    )
    existing = [path for path in candidates if path.exists()]
    if existing:
        return existing[0]
    raise FileNotFoundError("exp297 config.yaml was not found")


def get_nested(mapping: Mapping[str, Any], dotted_key: str) -> Any:
    current: Any = mapping
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def validate_fail_closed(config: Mapping[str, Any]) -> None:
    exact = {
        "experiment.route": "pf_beam",
        "experiment.inference_enabled": False,
        "execution.inference": False,
        "execution.submission": False,
        "execution.kaggle_inference_push_approved": False,
        "inference.enabled": False,
        "inference.create_submission": False,
        "candidate_bank.persist_selected_row_prediction": False,
        "audit.truth_readout.persist_truth_joined_candidate_rows": False,
    }
    mismatches = {
        key: {"expected": expected, "actual": get_nested(config, key)}
        for key, expected in exact.items()
        if get_nested(config, key) != expected
    }
    if mismatches:
        raise ValueError(f"exp297 inference stop contract mismatch: {mismatches}")


def stop_inference() -> None:
    config = yaml.safe_load(find_config_path().read_text()) or {}
    validate_fail_closed(config)
    raise RuntimeError(
        "exp297 is a train-side Stage-2 evidence audit: raw-test inference and "
        "submission generation are forbidden by the fixed contract"
    )


validate_fail_closed(yaml.safe_load(find_config_path().read_text()) or {})

if EXECUTE_NOTEBOOK:
    stop_inference()
