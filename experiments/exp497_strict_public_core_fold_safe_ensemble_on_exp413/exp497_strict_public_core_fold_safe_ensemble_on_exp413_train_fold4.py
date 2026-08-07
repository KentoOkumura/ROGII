# %% [markdown]
# # exp497 Stage M — outer fold 4
#
# Stage Pのtarget-free surfaceを読み、outer-valid fold 4を完全holdoutして2 branchの
# inner-4 OOF stackを学習する。LGB 24 + CatBoost 16 = 40 boosters、Ridge 2、
# exp413 retraining 0を固定する。

# %% [markdown]
# ## Contents
# 1. Imports and configuration
# 2. Input contracts and previews
# 3. Fold-safe Stage M execution
# 4. Metrics and artifacts

# %%
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml
from IPython.display import display

from src.strict_public_core import (
    find_artifact,
    find_competition_train_dir,
    run_stage_m_outer,
    sha256_file,
    training_inventory,
)

OUTER_FOLD = 4
EXPERIMENT = "exp497_strict_public_core_fold_safe_ensemble_on_exp413"
ROOT = Path.cwd()
EXPERIMENT_DIR = ROOT / "experiments" / EXPERIMENT
if not EXPERIMENT_DIR.is_dir():
    EXPERIMENT_DIR = ROOT
CONFIG = yaml.safe_load((EXPERIMENT_DIR / "config.yaml").read_text())
OUTPUT_DIR = (
    Path("/kaggle/working/artifacts")
    if Path("/kaggle/working").exists()
    else EXPERIMENT_DIR / "artifacts"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# %% [markdown]
# ## 2. Input contracts and previews
# %%
inventory = training_inventory()
print("experiment", EXPERIMENT, "route", CONFIG["experiment"]["route"])
print("outer_fold", OUTER_FOLD, "global_inventory", json.dumps(inventory, indent=2))
print("this shard", {"lgb": 24, "cat": 16, "boosters": 40, "ridge": 2, "exp413_retraining": 0})
feature_paths = [
    find_artifact(f"stage_p_fold{fold}_physical_features.parquet") for fold in range(5)
]
summary_paths = [find_artifact(f"stage_p_fold{fold}_summary.json") for fold in range(5)]
parent_path = find_artifact("stage_d_oof_predictions.parquet")
runtime_path = find_artifact("public_notebook_replay_audit.py")
train_dir = find_competition_train_dir()
print("stage_p", [(path.name, sha256_file(path)) for path in feature_paths])
print("parent", parent_path, sha256_file(parent_path))
display(
    pd.read_parquet(feature_paths[OUTER_FOLD]).head(3)[
        ["id", "well", "outer_fold", "target", "likpf_scale_5"]
    ]
)
# %% [markdown]
# ## 3. Fold-safe Stage M execution
# %%
summary = run_stage_m_outer(
    outer_fold=OUTER_FOLD,
    output_dir=OUTPUT_DIR,
    stage_p_feature_paths=feature_paths,
    stage_p_summary_paths=summary_paths,
    parent_oof_path=parent_path,
    parent_oof_sha256=CONFIG["data"]["parent_exp413"]["expected_final_oof_sha256"],
    train_dir=train_dir,
    public_runtime_path=runtime_path,
)
# %% [markdown]
# ## 4. Metrics and artifacts
# %%
if summary["fitted_boosters"] != 40 or summary["inventory"]["exp413_retraining"] != 0:
    raise RuntimeError("Stage M fold cost contract failed")
display(pd.DataFrame([summary["metrics"]]))
display(pd.DataFrame([summary["weights"]]))
print(json.dumps(summary, indent=2))
print("Stage M outer fold 4 COMPLETE. Inference and submission were not generated.")
