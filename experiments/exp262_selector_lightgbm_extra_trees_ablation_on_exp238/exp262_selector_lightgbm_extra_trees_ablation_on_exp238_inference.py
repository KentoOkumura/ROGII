# %% [markdown]
# # exp262 selector LightGBM extra-trees ablation on exp238 inference
#
# The initial experiment is selector-only. Raw-test inference and downstream
# exp218 retraining require a passed selector guard and a separate user approval.

# %% [markdown]
# ## Contents
# 1. Imports and configuration
# 2. Train-side guard inspection
# 3. Intentional inference stop

# %%
from __future__ import annotations

import json
from pathlib import Path

import yaml

PACKAGE_DIR = Path.cwd()
if not (PACKAGE_DIR / "config.yaml").exists():
    PACKAGE_DIR = Path(
        "experiments/exp262_selector_lightgbm_extra_trees_ablation_on_exp238"
    )
CONFIG = yaml.safe_load((PACKAGE_DIR / "config.yaml").read_text())
OUTPUT_PREFIX = str(CONFIG["audit"]["output_prefix"])
print(
    json.dumps(
        {
            "experiment": CONFIG["experiment"]["name"],
            "route": CONFIG["experiment"]["route"],
            "inference_mode": CONFIG["inference"]["mode"],
            "inference_enabled": CONFIG["inference"]["enabled"],
            "submission_enabled": CONFIG["inference"]["submission"],
            "selector_training_during_inference": False,
            "downstream_training_during_inference": False,
        },
        indent=2,
    )
)

# %% [markdown]
# ## 2. Train-side guard inspection

# %%
guard_name = f"{OUTPUT_PREFIX}_guard.json"
guard_candidates = [PACKAGE_DIR / "artifacts" / guard_name]
if Path("/kaggle/input").exists():
    guard_candidates.extend(Path("/kaggle/input").rglob(guard_name))
guard_path = next((path for path in guard_candidates if path.exists()), None)
if guard_path is not None:
    guard = json.loads(guard_path.read_text())
    print(json.dumps({"guard_path": str(guard_path), "guard": guard}, indent=2))
else:
    print(json.dumps({"guard_path": None, "guard": "not_available"}, indent=2))

# %% [markdown]
# ## 3. Intentional inference stop

# %%
if not bool(CONFIG["inference"]["enabled"]):
    raise RuntimeError(
        "exp262 initial scope is selector-only. Even if the train-side guard passes, "
        "raw-test inference or downstream exp218 retraining needs a separate design and "
        "explicit user approval in the same experiment."
    )
raise RuntimeError("Inference implementation is intentionally absent from the initial scope.")
