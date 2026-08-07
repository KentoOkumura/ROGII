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
# # exp256 near-bucket attribution — inference guard
#
# 本実験は exp250 保存済み train-side 診断だけを読む。raw-test corridor、candidate、
# feature、prediction、submission は生成しない。

# %% [markdown]
# ## Contents
#
# 1. Imports and configuration
# 2. Train-side-only fail-closed guard

# %% [markdown]
# ## 1. Imports and configuration

# %%
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

EXPERIMENT_NAME = "exp256_segment_local_corridor_near_bucket_signal_attribution_readout"
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
        Path("/kaggle/working/config.yaml"),
    ):
        if candidate.exists():
            return candidate
    raise FileNotFoundError("exp256 config.yaml was not found")


with find_config_path().open() as fp:
    config = yaml.safe_load(fp) or {}

print("Experiment:", get_nested(config, "experiment.name"))
print("Route:", get_nested(config, "experiment.route"))
print("Inference mode:", get_nested(config, "inference.mode"))
print("Submission enabled:", get_nested(config, "inference.create_submission"))

# %% [markdown]
# ## 2. Train-side-only fail-closed guard

# %%
assert get_nested(config, "experiment.name") == EXPERIMENT_NAME
assert get_nested(config, "experiment.route") == "pf_beam"
assert get_nested(config, "inference.enabled") is False
assert get_nested(config, "inference.create_submission") is False
assert get_nested(config, "inference.selected_candidate") is None
assert get_nested(config, "model.pf_beam_regeneration_count") == 0
assert get_nested(config, "model.corridor_regeneration_count") == 0
raise RuntimeError(
    "exp256 inference is disabled: this is a fixed exp250 train-side attribution "
    "readout and must not create raw-test candidates, predictions, features, or submission.csv."
)
