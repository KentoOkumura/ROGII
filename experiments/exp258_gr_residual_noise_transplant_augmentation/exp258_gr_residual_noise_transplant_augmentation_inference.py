# %% [markdown]
# # exp258 inference guard

# %% [markdown]
# ## Contents
# 1. Runtime contract
# 2. Selector and final guard verification
# 3. Clean current-test inference contract

# %%
from __future__ import annotations

import json
from pathlib import Path

import yaml

PACKAGE_DIR = Path.cwd()
if not (PACKAGE_DIR / "config.yaml").exists():
    PACKAGE_DIR = Path("experiments/exp258_gr_residual_noise_transplant_augmentation")
CONFIG = yaml.safe_load((PACKAGE_DIR / "config.yaml").read_text())
OUTPUT_PREFIX = str(CONFIG["experiment"]["name"])

print(
    json.dumps(
        {
            "experiment": OUTPUT_PREFIX,
            "inference_enabled": CONFIG["execution"]["inference_enabled"],
            "augmentation_at_inference": CONFIG["inference"]["augmentation_at_inference"],
            "selected_model": CONFIG["inference"]["selected_model"],
            "submission_requested": CONFIG["inference"]["submission_requested"],
        },
        indent=2,
    )
)

# %% [markdown]
# ## 2. Selector and final guard verification

# %%
if CONFIG["execution"]["inference_enabled"] is not True:
    raise RuntimeError(
        "Inference is intentionally disabled until both selector and final OOF guards pass."
    )
if CONFIG["inference"]["augmentation_at_inference"] is not False:
    raise RuntimeError("GR residual augmentation is train-only and forbidden at inference")
if CONFIG["inference"]["selected_model"] is None:
    raise RuntimeError("No final model has been selected after the OOF guard")

selector_summary_name = f"{OUTPUT_PREFIX}_selector_summary.json"
final_summary_name = f"{OUTPUT_PREFIX}_final_summary.json"
input_root = Path("/kaggle/input")
selector_matches = list(input_root.rglob(selector_summary_name)) if input_root.exists() else []
final_matches = list(input_root.rglob(final_summary_name)) if input_root.exists() else []
if not selector_matches or not final_matches:
    raise FileNotFoundError("selector/final summaries are required for inference")
selector_summary = json.loads(selector_matches[0].read_text())
final_summary = json.loads(final_matches[0].read_text())
if selector_summary.get("decision", {}).get("guard_pass") is not True:
    raise RuntimeError("selector guard is not passing")
if final_summary.get("decision", {}).get("guard_pass") is not True:
    raise RuntimeError("final OOF guard is not passing")
if selector_summary.get("variant") != CONFIG["execution"]["primary_variant"]:
    raise RuntimeError("inference selector variant is not the primary residual-block variant")

# %% [markdown]
# ## 3. Clean current-test inference contract
#
# This experiment intentionally does not generate a submission before the two
# train-side guards pass. After selection, port the exp238 parity-safe current-test
# generator without changing its raw GR inputs, load exactly 20 saved augmented-
# trained selector models and 15 saved final LightGBM models, create the same 35
# fold-matched rank-slot features, and average the 15 final TVT predictions.
# Residual donors, true TVT, synthetic GR, and selector/final retraining are all
# forbidden in this notebook.

# %%
raise RuntimeError(
    "Guards passed, but the saved-model current-test port has not yet been activated. "
    "This explicit stop prevents accidental submission from a train-only implementation."
)
