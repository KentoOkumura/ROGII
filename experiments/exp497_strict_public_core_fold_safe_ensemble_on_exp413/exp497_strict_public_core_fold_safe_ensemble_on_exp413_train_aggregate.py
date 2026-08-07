# %% [markdown]
# # exp497 Stage E — meta-fold blend and promotion gate
#
# 5 outer-fold shardのstrict public-core OOFをfreeze確認後に結合し、保存済みexp413
# OOFとleave-one-outer-fold-outのconstant convex blendを作る。全AND gateがFAILなら
# selected OOFをexp413へ戻し、inference/submissionは生成しない。

# %% [markdown]
# ## Contents
# 1. Imports and configuration
# 2. Stage M and parent contracts
# 3. Meta-fold blend and fixed gate
# 4. Results and stop

# %%
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml
from IPython.display import display

from src.strict_public_core import find_artifact, run_stage_e, sha256_file

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
# ## 2. Stage M and parent contracts

# %%
prediction_paths = [find_artifact(f"stage_m_outer{fold}_predictions.parquet") for fold in range(5)]
summary_paths = [find_artifact(f"stage_m_outer{fold}_summary.json") for fold in range(5)]
parent_path = find_artifact("stage_d_oof_predictions.parquet")
hidden_path = find_artifact("exp115_hidden_like_spatial_holdout_from_ppt_fold_assignments.csv")
print("stage_m", [(path.name, sha256_file(path)) for path in prediction_paths])
print("parent", parent_path, sha256_file(parent_path))
print("hidden", hidden_path, sha256_file(hidden_path))
display(
    pd.DataFrame([json.loads(path.read_text()) for path in summary_paths])[
        ["outer_fold", "rows", "wells", "fitted_boosters", "elapsed_seconds"]
    ]
)

# %% [markdown]
# ## 3. Meta-fold blend and fixed gate

# %%
summary = run_stage_e(
    output_dir=OUTPUT_DIR,
    stage_m_prediction_paths=prediction_paths,
    stage_m_summary_paths=summary_paths,
    parent_oof_path=parent_path,
    parent_oof_sha256=CONFIG["data"]["parent_exp413"]["expected_final_oof_sha256"],
    hidden_like_assignment_path=hidden_path,
    hidden_like_assignment_sha256=CONFIG["data"]["hidden_like_assignment"]["expected_sha256"],
    public_core_weight_bounds=tuple(CONFIG["ensemble"]["public_core_weight_bounds"]),
)

# %% [markdown]
# ## 4. Results and stop

# %%
display(pd.DataFrame(summary["weights"]))
display(pd.DataFrame([summary["pooled"]]))
display(pd.DataFrame([summary["promotion_gate"]["checks"]]))
print(json.dumps(summary, indent=2))
if summary["promotion_gate"]["passed"]:
    print(
        "Promotion gate PASS. Stop: inference implementation/run still requires "
        "a separate decision."
    )
else:
    print(
        "Promotion gate FAIL. exp413 remains selected; no same-OOF rescue, "
        "inference, or submission."
    )
