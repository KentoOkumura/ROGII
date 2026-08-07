# %% [markdown]
# # exp278 formation gradient prefix stability risk readout on exp273 — inference
#
# Diagnostic-only experiment. Raw-test inference, gate application, and submission are out of scope.

# %% [markdown]
# ## Contents
# 1. Disabled inference contract
# 2. Scope and next decision

# %%
from pathlib import Path

import yaml
from IPython import get_ipython
from IPython.display import display

EXPERIMENT_NAME = "exp278_formation_gradient_prefix_stability_risk_readout_on_exp273"
PACKAGE_DIR = Path.cwd()
CONFIG_CANDIDATES = [
    PACKAGE_DIR / "config.yaml",
    PACKAGE_DIR / "experiments" / EXPERIMENT_NAME / "config.yaml",
]
CONFIG_PATH = next((path for path in CONFIG_CANDIDATES if path.is_file()), None)
if CONFIG_PATH is None:
    raise FileNotFoundError(f"Could not locate config.yaml; checked={CONFIG_CANDIDATES}")
CONFIG = yaml.safe_load(CONFIG_PATH.read_text())
EXECUTE_NOTEBOOK = get_ipython() is not None


# %% [markdown]
# ## 1. Disabled inference contract

# %%
if EXECUTE_NOTEBOOK:
    contract = {
        "experiment": CONFIG["experiment"]["name"],
        "route": CONFIG["experiment"]["route"],
        "inference_enabled": bool(CONFIG["execution"]["inference_enabled"]),
        "submission_enabled": bool(CONFIG["execution"]["submission_enabled"]),
        "hmm_paths_generated": int(CONFIG["execution"]["hmm_paths_generated"]),
    }
    display(contract)
    assert contract == {
        "experiment": EXPERIMENT_NAME,
        "route": "pf_beam",
        "inference_enabled": False,
        "submission_enabled": False,
        "hmm_paths_generated": 0,
    }


# %% [markdown]
# ## 2. Scope and next decision
#
# exp278 は train-side readout のみ。guard PASSでも、このexperiment内でhard gateやraw-test pathを
# 作らない。別のfold-safe gate auditを新規設計し、明示承認を得るまでinferenceは無効のままとする。

# %%
if EXECUTE_NOTEBOOK:
    print("Diagnostic-only inference contract confirmed; no submission is generated.")
