# %% [markdown]
# # exp497 Stage P — outer fold 0 physical feature shard
#
# Kaggle CPUでouter fold 0だけの独立LikPF/Beam候補を生成する。単一kernel 5-fold実行が
# 12時間上限に達したため、seed・粒子数・特徴契約を変えず実行単位だけを分割する。

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
    run_stage_p_shard,
    sha256_file,
)

SHARD_FOLD = 0
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

# %%
print("experiment", EXPERIMENT, "route", CONFIG["experiment"]["route"], "stage_p_fold", SHARD_FOLD)
print(
    "particles/seeds",
    CONFIG["public_core"]["pf"]["particles"],
    CONFIG["public_core"]["pf"]["seeds"],
)
print("models/parent_retraining", 0, CONFIG["training_contract"]["exp413_parent_retraining"])
exp072_path = find_artifact(
    "exp063_full_replay_feature_cache_pixiux_likpf_public_replay_train_features.csv.gz"
)
parent_path = find_artifact("stage_d_oof_predictions.parquet")
runtime_path = find_artifact("public_notebook_replay_audit.py")
train_dir = find_competition_train_dir()
print("exp072", exp072_path, sha256_file(exp072_path))
print("parent", parent_path, sha256_file(parent_path))
print("runtime", runtime_path, sha256_file(runtime_path))

# %%
summary = run_stage_p_shard(
    shard_fold=SHARD_FOLD,
    output_dir=OUTPUT_DIR,
    exp072_cache_path=exp072_path,
    exp072_cache_sha256=CONFIG["data"]["exp072_public_base"]["expected_sha256"],
    parent_oof_path=parent_path,
    parent_oof_sha256=CONFIG["data"]["parent_exp413"]["expected_final_oof_sha256"],
    train_dir=train_dir,
    public_runtime_path=runtime_path,
    particles=int(CONFIG["public_core"]["pf"]["particles"]),
    seeds=int(CONFIG["public_core"]["pf"]["seeds"]),
)
if summary["shard_fold"] != SHARD_FOLD or summary["status"] != "complete":
    raise RuntimeError("Stage P shard completion contract failed")
display(pd.DataFrame([summary])[["shard_fold", "rows", "wells", "elapsed_seconds"]])
print(json.dumps(summary, indent=2))
