# %% [markdown]
# # exp503 exp490 strength / weakness and prefix policy readout — inference guard
#
# exp503 is a train-OOF diagnostic. It must not create test predictions or a
# submission. A future masked-prefix replay requires a separate approved design.

# %%
from pathlib import Path

import yaml

EXPERIMENT_NAME = "exp503_exp490_strength_weakness_prefix_policy_readout"
PACKAGE_DIR = Path.cwd()


def find_config() -> Path:
    for root in (PACKAGE_DIR, *PACKAGE_DIR.parents):
        candidate = root / "experiments" / EXPERIMENT_NAME / "config.yaml"
        if candidate.is_file():
            return candidate
    candidate = Path("/kaggle/working/config.yaml")
    if candidate.is_file():
        return candidate
    raise FileNotFoundError("exp503 config.yaml was not found")


# %%
with find_config().open("r", encoding="utf-8") as handle:
    CONFIG = yaml.safe_load(handle)

if CONFIG["implementation"]["inference_enabled"]:
    raise RuntimeError("exp503 inference must remain disabled")
if CONFIG["implementation"]["submission_enabled"]:
    raise RuntimeError("exp503 submission must remain disabled")

print("exp503 is diagnostic-only: inference and submission are disabled.")
