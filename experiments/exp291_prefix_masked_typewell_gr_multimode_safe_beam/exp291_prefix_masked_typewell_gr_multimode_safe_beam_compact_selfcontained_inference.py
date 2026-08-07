# %% [markdown]
# # exp291 prefix-masked Type Well GR multimode safe beam inference
#
# exp291 is a train-side masked backtest. It deliberately has no raw-test
# decoder, prediction, submission, or inference artifact contract.

# %% [markdown]
# ## Contents
# 1. Imports and runtime helpers
# 2. Configuration lookup
# 3. Inference contract validation
# 4. Disabled inference report

# %% [markdown]
# ## 1. Imports and runtime helpers

# %%
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

EXPERIMENT_NAME = "exp291_prefix_masked_typewell_gr_multimode_safe_beam"


def project_root() -> Path:
    start = Path.cwd()
    for candidate in (start, *start.parents):
        if (candidate / "project.yml").exists():
            return candidate
    return start


# %% [markdown]
# ## 2. Configuration lookup

# %%
def load_config() -> dict[str, Any]:
    root = project_root()
    candidates = (
        Path.cwd() / "config.yaml",
        root / "experiments" / EXPERIMENT_NAME / "config.yaml",
    )
    for path in candidates:
        if not path.exists():
            continue
        value = yaml.safe_load(path.read_text()) or {}
        if value.get("experiment", {}).get("name") == EXPERIMENT_NAME:
            return value
    raise FileNotFoundError(f"exp291 config not found in {[str(path) for path in candidates]}")


# %% [markdown]
# ## 3. Inference contract validation

# %%
CONFIG = load_config()
INFERENCE = CONFIG.get("inference", {})
EXECUTION = CONFIG.get("execution", {})
if bool(INFERENCE.get("enabled")):
    raise ValueError("exp291 fixed contract requires inference.enabled=false")
if bool(INFERENCE.get("create_submission")):
    raise ValueError("exp291 must not create a submission")
if bool(EXECUTION.get("inference")) or bool(EXECUTION.get("submission")):
    raise ValueError("exp291 execution contract forbids inference and submission")


# %% [markdown]
# ## 4. Disabled inference report

# %%
REPORT = {
    "experiment": EXPERIMENT_NAME,
    "route": CONFIG.get("experiment", {}).get("route"),
    "inference_enabled": False,
    "submission_enabled": False,
    "reason": "fixed known-prefix train-side diagnostic; decoder requires separate approval",
}
print(json.dumps(REPORT, indent=2, sort_keys=True))
