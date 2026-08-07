# %% [markdown]
# # exp295 prefix-anchored whole-well GR alignment SSM inference
#
# Stage C is intentionally unavailable.  A current-test decoder may be added
# to this experiment only after Stage A and the full five-fold Stage B pass all
# fixed gates and the user separately approves inference implementation.

# %% [markdown]
# ## Contents
# 1. Imports and path helpers
# 2. Configuration lookup
# 3. Fail-closed Stage C contract
# 4. Disabled inference report

# %% [markdown]
# ## 1. Imports and path helpers

# %%
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

EXPERIMENT_NAME = "exp295_prefix_anchored_wholewell_gr_alignment_ssm"


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
    raise FileNotFoundError(f"exp295 config not found in {[str(path) for path in candidates]}")


# %% [markdown]
# ## 3. Fail-closed Stage C contract

# %%
def validate_disabled_inference(config: dict[str, Any]) -> dict[str, Any]:
    inference = config.get("inference", {})
    execution = config.get("execution", {})
    if bool(inference.get("enabled")):
        raise ValueError("exp295 Stage C inference is not approved")
    if bool(inference.get("create_submission")):
        raise ValueError("exp295 must not create a submission")
    if bool(execution.get("inference_approved")):
        raise ValueError("execution.inference_approved must remain false before Stage B promotion")
    if bool(execution.get("submission_approved")):
        raise ValueError("execution.submission_approved must remain false")
    stage_b = execution.get("stage_b_plan", {})
    stage_c = execution.get("stage_c_plan", {})
    if stage_c.get("prerequisite") != "stage_b_lb5x_promotion_pass":
        raise ValueError("Stage C prerequisite contract changed")
    return {
        "experiment": EXPERIMENT_NAME,
        "route": config.get("experiment", {}).get("route"),
        "inference_enabled": False,
        "create_submission": False,
        "stage_b_prerequisite": stage_b.get("prerequisite"),
        "stage_c_prerequisite": stage_c.get("prerequisite"),
        "reason": inference.get("reason"),
    }


# %% [markdown]
# ## 4. Disabled inference report

# %%
CONFIG = load_config()
REPORT = validate_disabled_inference(CONFIG)
print(json.dumps(REPORT, indent=2, sort_keys=True))
