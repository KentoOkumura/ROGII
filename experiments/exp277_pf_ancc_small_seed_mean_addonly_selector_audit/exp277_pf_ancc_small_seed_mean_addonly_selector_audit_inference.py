# %% [markdown]
# # exp277 PF ANCC small-seed mean add-only selector audit — inference
#
# この実験はtrain-side selector/downstream audit専用である。raw-test PF再生成、
# inference、submissionはguard通過後の別承認まで禁止する。

# %% [markdown]
# ## Contents
# 1. Configuration guard
# 2. Disabled inference contract

# %% [markdown]
# ## 1. Configuration guard

# %%
from pathlib import Path

import yaml

EXPERIMENT_NAME = "exp277_pf_ancc_small_seed_mean_addonly_selector_audit"
PACKAGE_DIR = Path.cwd()
if not (PACKAGE_DIR / "config.yaml").exists():
    PACKAGE_DIR = Path("experiments") / EXPERIMENT_NAME
CONFIG = yaml.safe_load((PACKAGE_DIR / "config.yaml").read_text())
assert CONFIG["experiment"]["route"] == "ensemble"
assert CONFIG["inference"]["enabled"] is False
assert CONFIG["inference"]["create_submission"] is False
assert CONFIG["inference"]["submit_to_kaggle"] is False

# %% [markdown]
# ## 2. Disabled inference contract

# %%
raise RuntimeError(
    "exp277 is train-side only: raw-test PF regeneration, inference, and submission are disabled"
)
