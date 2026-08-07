# %% [markdown]
# # exp300 inference: not applicable
#
# exp300 は保存済み OOF の診断専用実験である。current-test予測、submission、
# candidate再生成、model inferenceは設計範囲外であり、この notebook は禁止契約を表示するだけとする。

# %% [markdown]
# ## Contents
# 1. Configuration check
# 2. Inference prohibition contract

# %%
from __future__ import annotations

from pathlib import Path

import yaml


EXPERIMENT_NAME = "exp300_exp264_vs_exp274_well_selector_readout"


def find_repo_root() -> Path:
    for candidate in [Path.cwd(), *Path.cwd().parents]:
        if (candidate / "project.yml").exists() and (candidate / "experiments").exists():
            return candidate
    raise FileNotFoundError("ROGII repository root was not found")


ROOT = find_repo_root()
EXPERIMENT_DIR = ROOT / "experiments" / EXPERIMENT_NAME
config = yaml.safe_load((EXPERIMENT_DIR / "config.yaml").read_text())
print("Experiment:", EXPERIMENT_NAME)
print("Status:", config["experiment"]["status"])
print("Trained boosters:", config["model"]["params"]["trained_boosters"])

# %% [markdown]
# ## 2. Inference prohibition contract

# %%
assert config["model"]["params"]["trained_boosters"] == 0
assert config["runtime"]["kaggle"]["inference_run_on_push"] is False
print("No inference or submission is generated for this diagnostic experiment.")
print("Use result.md and artifacts/selector_readout_summary.json as the handoff.")

