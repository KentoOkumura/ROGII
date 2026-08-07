# %% [markdown]
# # exp496 exp486 absolute-geometry fixed13 selector on exp264 — inference
#
# Inference is intentionally disabled. The train-side selector must first pass
# its fixed all-AND gate, after which current-test exp486 candidate generation,
# downstream TVT integration, inference, and submission each require a new
# approved design. Selector scores alone are not a TVT submission surface.

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

EXPERIMENT_NAME = "exp496_exp486_absolute_geometry_fixed13_selector_on_exp264"


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
    raise FileNotFoundError("exp496 config.yaml did not resolve")


CONFIG = yaml.safe_load(find_config().read_text())
print(
    json.dumps(
        {
            "experiment": EXPERIMENT_NAME,
            "route": CONFIG["experiment"]["route"],
            "train_run_approved": CONFIG["execution"]["run_approved"],
            "train_stage": CONFIG["execution"]["stage"],
            "inference_enabled": CONFIG["experiment"]["inference_enabled"],
            "downstream_tvt_enabled": CONFIG["model"]["downstream_tvt_stage"]["enabled"],
            "current_test_exp486_candidate_generated": False,
            "submission_enabled": CONFIG["execution"]["submission"],
        },
        indent=2,
    )
)

# %% [markdown]
# ## 2. Fail-closed inference boundary

# %%
raise RuntimeError(
    "exp496 inference is disabled: train-side Stage A/C has not been run and "
    "passed, current-test exp486 absolute-geometry candidate generation is "
    "not implemented or approved, downstream TVT is disabled, and no "
    "submission may be generated from selector scores alone."
)
