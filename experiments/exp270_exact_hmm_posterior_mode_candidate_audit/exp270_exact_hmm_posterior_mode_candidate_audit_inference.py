# %% [markdown]
# # exp270 exact HMM posterior mode candidate audit inference
#
# exp270 is deliberately train-side diagnostic only. This notebook exists to
# make the disabled inference/submission contract explicit and fail closed.

# %% [markdown]
# ## Contents
# 1. Imports
# 2. Load and display the experiment contract
# 3. Fail-closed inference guard

# %%
from __future__ import annotations

import json
from pathlib import Path

import yaml

EXPERIMENT_NAME = "exp270_exact_hmm_posterior_mode_candidate_audit"


def project_root() -> Path:
    start = Path.cwd()
    for candidate in (start, *start.parents):
        if (candidate / "project.yml").exists():
            return candidate
    return start


def load_config() -> dict:
    candidates = (
        Path.cwd() / "config.yaml",
        project_root() / "experiments" / EXPERIMENT_NAME / "config.yaml",
    )
    for path in candidates:
        if not path.exists():
            continue
        config = yaml.safe_load(path.read_text()) or {}
        if config.get("experiment", {}).get("name") == EXPERIMENT_NAME:
            return config
    raise FileNotFoundError(f"exp270 config not found in {[str(path) for path in candidates]}")


# %% [markdown]
# ## 2. Load and display the experiment contract

# %%
CONFIG = load_config()
CONTRACT = {
    "experiment": EXPERIMENT_NAME,
    "route": CONFIG["experiment"]["route"],
    "parent": CONFIG["lineage"]["parent"],
    "inference_enabled": CONFIG["inference"]["enabled"],
    "submission_creation": CONFIG["inference"]["create_submission"],
    "selected_candidate": CONFIG["inference"]["selected_candidate"],
    "reason": "train-side posterior-mode and oracle-headroom audit only",
}
print(json.dumps(CONTRACT, indent=2, sort_keys=True))


# %% [markdown]
# ## 3. Fail-closed inference guard

# %%
if CONFIG["inference"]["enabled"] or CONFIG["inference"]["create_submission"]:
    raise RuntimeError("exp270 config unexpectedly enables inference or submission")

raise RuntimeError(
    "exp270 intentionally has no inference/submission path. Review the train-side "
    "candidate audit before creating any follow-up inference experiment."
)
