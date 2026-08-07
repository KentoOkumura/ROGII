# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
# kernelspec:
#   display_name: Python 3
#   language: python
#   name: python3
# ---
# %% [markdown]
# # exp223 joint_typewell_self_gr_hmm_likelihood_probe inference

# %% [markdown]
# This experiment is train-side only until the OOF, hidden-like, worst-well, and
# self-GR agreement gates pass. The inference notebook records that no raw-test
# prediction policy is selected.

# %%
from __future__ import annotations

import json
from typing import Any

from exact_hmm_smoother import to_jsonable
from settings import ExperimentPaths, get_nested, load_config


def print_json(title: str, payload: dict[str, Any]) -> None:
    print(f"\n## {title}")
    print(json.dumps(to_jsonable(payload), indent=2, sort_keys=True))


paths = ExperimentPaths()
config = load_config()
inference = get_nested(config, "inference") or {}

print_json(
    "inference status",
    {
        "experiment": paths.experiment_name,
        "route": get_nested(config, "experiment.route"),
        "mode": inference.get("mode"),
        "selected_candidate": inference.get("selected_candidate"),
        "notes": inference.get("notes"),
    },
)

if inference.get("mode") != "not_applicable_train_side_readout_only":
    raise ValueError("exp223 inference notebook is disabled until train-side gates pass")

print(
    "No submission.csv is generated. If a candidate passes the train-side gates, "
    "raw-test-safe self-GR HMM regeneration must be implemented in this same experiment."
)
