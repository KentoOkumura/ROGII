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
# # exp251 raw-test-safe dual-objective candidate ranker — inference
#
# Inference is intentionally disabled. The train notebook may use raw test only for a
# target-free feature provenance/parity audit. It does not produce a submission.

# %% [markdown]
# ## Contents
#
# 1. Imports and runtime contract
# 2. Required promotion evidence
# 3. Stop guard

# %% [markdown]
# ## 1. Imports and runtime contract

# %%
from __future__ import annotations

from settings import ExperimentPaths, get_nested, load_config

paths = ExperimentPaths()
config = load_config()
paths.require_kaggle_runtime()
paths.ensure_output_dirs()

contract = {
    "route": get_nested(config, "experiment.route"),
    "status": get_nested(config, "experiment.status"),
    "inference_mode": get_nested(config, "inference.mode"),
    "submission": get_nested(config, "inference.submission"),
    "required_stage": "completed train_after_feature_audit with every guard passing",
}
print(contract)

# %% [markdown]
# ## 2. Required promotion evidence
#
# A later inference implementation requires all of the following:
#
# - exact 297-feature provenance audit and selected-schema SHA;
# - raw-test selected-feature decompressed-content SHA;
# - ten-model manifest and model SHA;
# - overall, 1000+, hidden-like, and worst-well train-side guards passing;
# - explicit user approval for raw-test inference;
# - a separate submit-check before any competition submission.

# %% [markdown]
# ## 3. Stop guard

# %%
assert get_nested(config, "inference.submission") is False
raise RuntimeError(
    "exp251 inference is disabled until train-side and raw-test parity guards pass and "
    "the user explicitly authorizes inference. No submission was produced."
)
