# %% [markdown]
# # exp261 LightGBM extra-trees ablation on exp218 inference
#
# Inference remains intentionally disabled until the train-side ablation passes
# every guard and the user explicitly approves an inference implementation.

# %% [markdown]
# ## Contents
# 1. Imports and configuration
# 2. Train-side adoption contract
# 3. Disabled inference boundary

# %% [markdown]
# ## 1. Imports and configuration

# %%
from __future__ import annotations

import json
from pathlib import Path

import yaml

PACKAGE_DIR = Path.cwd()
if not (PACKAGE_DIR / "config.yaml").exists():
    PACKAGE_DIR = Path(
        "experiments/exp261_lightgbm_extra_trees_ablation_on_exp218"
    )
CONFIG = yaml.safe_load((PACKAGE_DIR / "config.yaml").read_text())

print(
    json.dumps(
        {
            "experiment": CONFIG["experiment"],
            "route": CONFIG["experiment"]["route"],
            "parent": CONFIG["lineage"]["parent"],
            "inference": CONFIG["inference"],
        },
        indent=2,
    )
)

# %% [markdown]
# ## 2. Train-side adoption contract

# %%
summary_filename = (
    "exp261_lightgbm_extra_trees_ablation_on_exp218_summary.json"
)
summary_candidates = [
    PACKAGE_DIR / "artifacts" / summary_filename,
    Path("/kaggle/input/exp261-lightgbm-extra-trees-ablation-on-exp218-train")
    / summary_filename,
]
if Path("/kaggle/input").exists():
    summary_candidates.extend(Path("/kaggle/input").rglob(summary_filename))
summary_path = next((path for path in summary_candidates if path.exists()), None)

print(
    {
        "train_summary_found": summary_path is not None,
        "train_summary_path": str(summary_path) if summary_path else None,
        "required_next_decision": "train guard pass plus explicit inference approval",
    }
)

# %% [markdown]
# ## 3. Disabled inference boundary

# %%
raise RuntimeError(
    "Inference is intentionally not implemented for exp261. Complete the "
    "approved train plan, verify every train-side guard, and obtain explicit "
    "user approval before adding saved-booster current-test inference in this "
    "same experiment directory."
)
