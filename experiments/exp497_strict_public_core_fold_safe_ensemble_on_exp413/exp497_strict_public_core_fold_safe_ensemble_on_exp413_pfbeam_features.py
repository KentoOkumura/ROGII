# %% [markdown]
# # exp497 Stage P — strict public-core physical feature shards
#
# Kaggle CPUで、公開source由来の195列base surfaceへ独立selector/learned LikPF
# bankと14-beam selector候補を追加する。truthは候補生成に使わず、outer foldごとに
# partitionして保存する。

# %% [markdown]
# ## Contents
# 1. Imports and runtime
# 2. Frozen configuration
# 3. Input resolution and previews
# 4. Stage P fold shards
# 5. Execution inventory and artifacts

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
# ## 2. Frozen configuration

# %%
print("experiment", CONFIG["experiment"]["name"])
print("route", CONFIG["experiment"]["route"])
print("stage", "stage_p_all_outer_fold_shards")
print("active_variants", CONFIG["public_core"]["active_variants"])
print("parent_retraining", CONFIG["training_contract"]["exp413_parent_retraining"])
print(
    "selector/learned PF multiplier",
    CONFIG["public_core"]["pf"]["selector_gr_sigma_multiplier"],
    CONFIG["public_core"]["pf"]["learned_gr_sigma_multiplier"],
)
print(
    "particles/seeds",
    CONFIG["public_core"]["pf"]["particles"],
    CONFIG["public_core"]["pf"]["seeds"],
)

# %% [markdown]
# ## 3. Input resolution and previews

# %%
exp072_path = find_artifact(
    "exp063_full_replay_feature_cache_pixiux_likpf_public_replay_train_features.csv.gz"
)
parent_path = find_artifact("stage_d_oof_predictions.parquet")
train_dir = find_competition_train_dir()
runtime_path = find_artifact("public_notebook_replay_audit.py")
print("exp072", exp072_path, sha256_file(exp072_path))
print("parent", parent_path, sha256_file(parent_path))
print("train_dir", train_dir)
print("runtime", runtime_path, sha256_file(runtime_path))
display(
    pd.read_csv(exp072_path, nrows=3)[["id", "well", "target", "last_known_tvt", "likpf_mean_d"]]
)

# %% [markdown]
# ## 4. Stage P fold shards

# %%
summaries = []
for shard_fold in range(5):
    summary = run_stage_p_shard(
        shard_fold=shard_fold,
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
    summaries.append(summary)
    display(pd.DataFrame([summary])[["shard_fold", "rows", "wells", "elapsed_seconds"]])

# %% [markdown]
# ## 5. Execution inventory and artifacts

# %%
aggregate = {
    "stage": "stage_p_all_outer_fold_shards",
    "status": "complete",
    "rows": sum(int(item["rows"]) for item in summaries),
    "wells": sum(int(item["wells"]) for item in summaries),
    "likelihood_pf_seed_banks": sum(
        int(item["physical_inventory"]["likelihood_pf_seed_banks"]) for item in summaries
    ),
    "seed_well_runs": sum(int(item["physical_inventory"]["seed_well_runs"]) for item in summaries),
    "particle_starts": sum(
        int(item["physical_inventory"]["particle_starts"]) for item in summaries
    ),
    "selector_beam_well_runs": sum(
        int(item["physical_inventory"]["selector_beam_well_runs"]) for item in summaries
    ),
    "shards": summaries,
}
if aggregate["rows"] != 3_783_989 or aggregate["wells"] != 773:
    raise RuntimeError("Stage P aggregate row/well contract failed")
if aggregate["particle_starts"] != 98_944_000:
    raise RuntimeError("Stage P particle-start contract failed")
(OUTPUT_DIR / "stage_p_summary.json").write_text(json.dumps(aggregate, indent=2) + "\n")
display(pd.DataFrame([{k: v for k, v in aggregate.items() if k != "shards"}]))
print("Stage P COMPLETE. No model, exp413 retraining, inference, or submission was run.")
