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
# # exp272 continuous well divergence risk readout — inference disabled
#
# exp272は保存済みtrain-side OOFだけを使う0-booster diagnosticであり、raw-test axis、
# selector、prediction、submissionを生成しない。

# %% [markdown]
# ## Contents
#
# 1. Imports and config
# 2. Disabled inference contract
# 3. Prohibited outputs

# %% [markdown]
# ## 1. Imports and config

# %%
from __future__ import annotations

import os

from settings import EXPERIMENT_NAME, load_config

EXECUTE_NOTEBOOK = os.environ.get("EXP272_IMPORT_ONLY", "0") != "1"


# %% [markdown]
# ## 2. Disabled inference contract

# %%
if EXECUTE_NOTEBOOK:
    config = load_config()
    assert config["experiment"]["name"] == EXPERIMENT_NAME
    assert config["experiment"]["route"] == "ensemble"
    assert config["execution"]["inference_enabled"] is False
    assert config["execution"]["submission_enabled"] is False
    assert config["inference"]["enabled"] is False
    assert config["inference"]["create_submission"] is False


# %% [markdown]
# ## 3. Prohibited outputs
#
# exp272のguardがPASSしても、別add-only候補の設計根拠になるだけで、この実験から
# current-test feature、prediction、submissionを作ってはならない。

# %%
if EXECUTE_NOTEBOOK:
    raise RuntimeError(
        "exp272 inference is intentionally disabled: train-side continuous risk readout only; "
        "no current-test features, predictions, or submission are allowed."
    )
