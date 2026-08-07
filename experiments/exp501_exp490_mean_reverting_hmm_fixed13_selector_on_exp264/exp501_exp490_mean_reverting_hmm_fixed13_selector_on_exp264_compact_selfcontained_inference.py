# %% [markdown]
# # exp501 exp490 mean-reverting HMM fixed13 selector on exp264 — inference
#
# Inference is intentionally not implemented. exp501 currently defines only a
# train-side OOF selector audit. Current-test exp490 generation, selector model
# deployment, downstream TVT, and submission require a separate frozen design
# and explicit approval.

# %% [markdown]
# ## Contents
# 1. Configuration and authorization state
# 2. Fail-closed inference boundary

# %% [markdown]
# ## 1. Configuration and authorization state

# %%
from __future__ import annotations

import json
from pathlib import Path

import yaml

EXPERIMENT_NAME = "exp501_exp490_mean_reverting_hmm_fixed13_selector_on_exp264"


def find_config() -> Path:
    candidates = [
        Path.cwd() / "config.yaml",
        Path.cwd() / "experiments" / EXPERIMENT_NAME / "config.yaml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = sorted(Path.cwd().rglob("config.yaml"))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"config.yaml did not resolve uniquely: {matches}")


CONFIG = yaml.safe_load(find_config().read_text())
STATE = {
    "experiment": EXPERIMENT_NAME,
    "route": CONFIG["experiment"]["route"],
    "train_implementation_complete": bool(CONFIG["implementation"]["code_created"]),
    "canonical_train_notebook_adopted": bool(
        CONFIG["implementation"]["canonical_train_notebook_adopted"]
    ),
    "run_approved": bool(CONFIG["execution"]["run_approved"]),
    "inference_implemented": bool(CONFIG["implementation"]["inference_implemented"]),
    "inference_enabled": bool(CONFIG["execution"]["inference"]),
    "submission_enabled": bool(CONFIG["execution"]["submission"]),
}
print(json.dumps(STATE, indent=2))

# %% [markdown]
# ## 2. Fail-closed inference boundary

# %%
raise RuntimeError(
    "exp501 inference is not implemented or approved. The train-side fixed13 "
    "selector audit does not authorize current-test exp490 regeneration, saved "
    "model deployment, downstream TVT, or submission."
)
