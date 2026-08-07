# %% [markdown]
# # exp276 exp264 compact tail-risk target-free gate audit — inference
#
# This experiment is a train-side, zero-booster diagnostic only. Current-test risk generation,
# fallback prediction, submission creation, and competition submission are intentionally disabled.

# %% [markdown]
# ## Contents
# 1. Imports and configuration
# 2. Disabled inference contract

# %% [markdown]
# ## 1. Imports and configuration

# %%
from __future__ import annotations

from pathlib import Path

import yaml
from IPython.display import display

EXPERIMENT_NAME = "exp276_exp264_compact_tail_risk_target_free_gate_audit"
PACKAGE_DIR = Path.cwd()
if not (PACKAGE_DIR / "config.yaml").exists():
    PACKAGE_DIR = Path("experiments") / EXPERIMENT_NAME
CONFIG = yaml.safe_load((PACKAGE_DIR / "config.yaml").read_text())


# %% [markdown]
# ## 2. Disabled inference contract
#
# A positive train-side audit would only justify designing a separately approved current-test port.
# It does not authorize selecting a favorable quantile, regenerating exp264 predictions, creating a
# submission, or submitting to the competition.

# %%
contract = {
    "experiment": CONFIG["experiment"]["name"],
    "route": CONFIG["experiment"]["route"],
    "inference_enabled": bool(CONFIG["execution"]["inference_enabled"]),
    "submission_enabled": bool(CONFIG["execution"]["submission_enabled"]),
    "create_submission": bool(CONFIG["inference"]["create_submission"]),
    "total_boosters": int(CONFIG["execution"]["total_boosters"]),
}
display(contract)
assert contract == {
    "experiment": EXPERIMENT_NAME,
    "route": "ensemble",
    "inference_enabled": False,
    "submission_enabled": False,
    "create_submission": False,
    "total_boosters": 0,
}
print("Inference and submission are disabled by the exp276 audit contract.")
