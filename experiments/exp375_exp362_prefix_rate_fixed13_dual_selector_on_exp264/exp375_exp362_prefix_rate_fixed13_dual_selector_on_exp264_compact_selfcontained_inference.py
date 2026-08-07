# %% [markdown]
# # exp375 exp362 prefix-rate fixed13 dual selector on exp264 — inference
#
# Inference is intentionally disabled until the Stage C integration gate passes
# and downstream TVT training is separately approved and completed. Exp362 has
# no current-test prefix-rate candidate or native-confidence artifact yet.

# %% [markdown]
# ## Contents
# 1. Configuration
# 2. Fail-closed inference boundary

# %%
from __future__ import annotations

import json
from pathlib import Path

import yaml

EXPERIMENT_NAME = "exp375_exp362_prefix_rate_fixed13_dual_selector_on_exp264"


def find_config() -> Path:
    candidates = [
        Path.cwd() / "config.yaml",
        Path.cwd() / "experiments" / EXPERIMENT_NAME / "config.yaml",
    ]
    for path in candidates:
        if path.exists():
            return path
    matches = sorted(Path.cwd().rglob("config.yaml"))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError("exp375 config.yaml did not resolve")


CONFIG = yaml.safe_load(find_config().read_text())
print(
    json.dumps(
        {
            "experiment": EXPERIMENT_NAME,
            "route": CONFIG["experiment"]["route"],
            "inference_enabled": CONFIG["experiment"]["inference_enabled"],
            "downstream_tvt_enabled": CONFIG["model"]["downstream_tvt_stage"][
                "enabled"
            ],
        },
        indent=2,
    )
)

# %% [markdown]
# ## 2. Fail-closed inference boundary

# %%
raise RuntimeError(
    "exp375 inference is disabled: Stage C must pass, then the fixed13 "
    "downstream TVT model and current-test prefix-rate exact-HMM candidate "
    "require separate approved runs. No submission may be generated from "
    "selector scores alone."
)
