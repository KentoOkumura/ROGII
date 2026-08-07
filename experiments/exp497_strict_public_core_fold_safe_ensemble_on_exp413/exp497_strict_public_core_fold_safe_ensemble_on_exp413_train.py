# %% [markdown]
# # exp497 strict public-core fold-safe ensemble — Kaggle train runbook
#
# この正規 train Notebook は、1回の Kaggle 実行に収まらない exp497 の Stage P →
# Stage M outer 0..4 → Stage E を固定順序で管理する。各 stage の実処理は同じ実験配下の
# Jupytext起点 shard Notebookが担い、このrunbookは承認、実行量、依存関係、入力契約を
# fail-closedで照合する。

# %% [markdown]
# ## Contents
# 1. Imports and experiment configuration
# 2. Frozen cost and route contract
# 3. Sharded notebook topology
# 4. Execution order and stop conditions

# %%
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml
from IPython.display import display

EXPERIMENT = "exp497_strict_public_core_fold_safe_ensemble_on_exp413"
ROOT = Path.cwd()
EXPERIMENT_DIR = ROOT / "experiments" / EXPERIMENT
if not EXPERIMENT_DIR.is_dir():
    EXPERIMENT_DIR = ROOT
CONFIG = yaml.safe_load((EXPERIMENT_DIR / "config.yaml").read_text())

# %% [markdown]
# ## 2. Frozen cost and route contract

# %%
training = CONFIG["training_contract"]
observed_cost = {
    "scientific_variants": int(training["scientific_variants"]),
    "ml_branches": int(training["ml_branches"]),
    "outer_folds": int(training["outer_folds"]),
    "inner_folds": int(training["inner_folds"]),
    "lightgbm_boosters": int(training["planned_lightgbm_boosters"]),
    "catboost_boosters": int(training["planned_catboost_boosters"]),
    "total_boosters": int(training["planned_total_boosters"]),
    "ridge_models": int(training["planned_ridge_models"]),
    "exp413_retraining": int(training["exp413_parent_retraining"]),
}
expected_cost = {
    "scientific_variants": 1,
    "ml_branches": 2,
    "outer_folds": 5,
    "inner_folds": 4,
    "lightgbm_boosters": 120,
    "catboost_boosters": 80,
    "total_boosters": 200,
    "ridge_models": 10,
    "exp413_retraining": 0,
}
if CONFIG["experiment"]["route"] != "ensemble":
    raise RuntimeError("exp497 route must remain ensemble")
if CONFIG["implementation"]["kaggle_run_approved"] is not True:
    raise RuntimeError("exp497 Kaggle execution is not approved")
if observed_cost != expected_cost:
    raise RuntimeError(f"Training cost contract changed: {observed_cost}")
display(pd.DataFrame([observed_cost]))

# %% [markdown]
# ## 3. Sharded notebook topology

# %%
stage_rows = [
    *[
        {
            "order": fold,
            "kind": f"pfbeam_features_fold{fold}",
            "accelerator": "Kaggle CPU",
            "depends_on": "exp072 + exp413 + raw train",
            "boosters": 0,
        }
        for fold in range(5)
    ],
    *[
        {
            "order": fold + 5,
            "kind": f"train_fold{fold}",
            "accelerator": "Kaggle GPU",
            "depends_on": "five Stage P shards + exp413 + raw train",
            "boosters": 40,
        }
        for fold in range(5)
    ],
    {
        "order": 10,
        "kind": "train_aggregate",
        "accelerator": "CPU",
        "depends_on": "five Stage M shards + exp413",
        "boosters": 0,
    },
]
topology = pd.DataFrame(stage_rows)
if int(topology["boosters"].sum()) != 200:
    raise RuntimeError("Shard booster sum must be exactly 200")
for kind in topology["kind"]:
    notebook = EXPERIMENT_DIR / f"{EXPERIMENT}_{kind}.ipynb"
    script = EXPERIMENT_DIR / f"{EXPERIMENT}_{kind}.py"
    if not notebook.is_file() or not script.is_file():
        raise FileNotFoundError(f"Missing shard source pair for {kind}")
display(topology)

# %% [markdown]
# ## 4. Execution order and stop conditions
#
# - Stage Pは12時間上限を避けるためouter fold別の5 CPU kernelで実行し、全SHAを確認する。
# - Stage Pの5 shard完了後、Kaggle GPUのStage Mをouter fold順に1本ずつ実行する。
# - 各Stage M shardはLGB 24 + CatBoost 16、Ridge 2、exp413再学習0を強制する。
# - 5 shardが揃った場合だけStage Eを実行する。
# - Stage Eの全AND gateがFAILならexp413 OOFを選択して停止する。PASSでもinferenceと
#   submissionは生成せず、同じexp497内での次段判断を待つ。
# - Colabはこの実行経路で使用しない。

# %%
print(
    json.dumps(
        {
            "experiment": EXPERIMENT,
            "status": CONFIG["experiment"]["status"],
            "route": CONFIG["experiment"]["route"],
            "execution": "Kaggle only; Stage M uses Kaggle GPU",
            "colab": False,
            "inference_enabled": CONFIG["experiment"]["inference_enabled"],
            "training_cost": observed_cost,
            "next_stage": "pfbeam_features_fold0..4",
        },
        indent=2,
    )
)
