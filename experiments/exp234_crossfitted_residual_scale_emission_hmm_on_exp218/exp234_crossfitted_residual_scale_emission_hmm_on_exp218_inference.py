# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # exp234_crossfitted_residual_scale_emission_hmm_on_exp218 inference
#
# この実験は train-side residual-scale guard と HMM readout 専用である。raw-test で
# fold-safe sigma を再生成する設計・検証は未実施のため、推論・提出を明示的に停止する。

# %% [markdown]
# ## Contents
#
# 1. Imports
# 2. Configuration and train-side guard
# 3. Inference stop condition

# %%
from __future__ import annotations

import json

from settings import ExperimentPaths, get_nested, load_config

# %% [markdown]
# ## 2. Configuration and train-side guard
#
# 実行可能な候補に昇格するには、train-side HMM の overall / bucket / hidden-like /
# worst-well guard に加え、hidden raw-test で residual scale を作る fold-safe protocol を
# 新たに設計して、ユーザー承認を得る必要がある。

# %%
paths = ExperimentPaths()
config = load_config()
inference_enabled = bool(get_nested(config, "experiment.inference_enabled"))
print(
    json.dumps(
        {
            "experiment": get_nested(config, "experiment.name"),
            "route": get_nested(config, "experiment.route"),
            "inference_enabled": inference_enabled,
            "reason": "raw-test residual-scale regeneration is intentionally out of scope",
            "artifacts_dir": str(paths.artifacts_dir),
        },
        indent=2,
        sort_keys=True,
    )
)


# %% [markdown]
# ## 3. Inference stop condition

# %%
if not inference_enabled:
    raise RuntimeError(
        "exp234 inference is disabled: do not regenerate sigma or submit until a later, "
        "explicitly approved raw-test protocol passes the train-side guards."
    )
