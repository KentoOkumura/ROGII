# %% [markdown]
# # exp392 exp389 Huber fixed13 dual selector on exp264 — inference
#
# Inference is intentionally disabled until the Stage C integration gate passes
# and downstream TVT training is separately approved and completed. Exp389 has
# no current-test Huber candidate or native-confidence artifact yet.

# %% [markdown]
# ## Contents
# 1. Configuration
# 2. Fail-closed inference boundary

# %%
from __future__ import annotations

import json
from pathlib import Path

import yaml

EXPERIMENT_NAME = "exp392_exp389_fixed13_dual_selector_on_exp264"


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
    raise FileNotFoundError("exp392 config.yaml did not resolve")


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
    "exp392 inference is disabled: Stage C must pass, then the fixed13 "
    "downstream TVT model and current-test Huber exact-HMM candidate "
    "require separate approved runs. No submission may be generated from "
    "selector scores alone."
)
