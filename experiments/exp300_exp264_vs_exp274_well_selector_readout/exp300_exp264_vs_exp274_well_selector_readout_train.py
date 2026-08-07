# %% [markdown]
# # exp300 exp264 vs exp274 well / selector readout
#
# 保存済み OOF だけを使う診断 notebook。モデル学習、候補再生成、推論、提出は行わない。

# %% [markdown]
# ## Contents
# 1. Imports and paths
# 2. Configuration and source contracts
# 3. Input checks
# 4. Readout orchestration
# 5. Well and long-tail results
# 6. Selector results
# 7. Reproducibility and non-use contract

# %%
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pandas as pd
import yaml
from IPython.display import display


EXPERIMENT_NAME = "exp300_exp264_vs_exp274_well_selector_readout"


def find_repo_root() -> Path:
    candidates = [Path.cwd(), *Path.cwd().parents]
    for candidate in candidates:
        if (candidate / "project.yml").exists() and (candidate / "experiments").exists():
            return candidate
    raise FileNotFoundError("ROGII repository root was not found")


ROOT = find_repo_root()
EXPERIMENT_DIR = ROOT / "experiments" / EXPERIMENT_NAME
ARTIFACT_DIR = EXPERIMENT_DIR / "artifacts"
SOURCE_DIR = ARTIFACT_DIR / "source_inputs"
RUN_FULL_READOUT = os.environ.get("EXP300_RUN_FULL_READOUT", "0") == "1"

print("Experiment:", EXPERIMENT_NAME)
print("Repository:", ROOT)
print("Route: ml_model (diagnostic only)")
print("Run full readout:", RUN_FULL_READOUT)

# %% [markdown]
# ## 2. Configuration and source contracts

# %%
config = yaml.safe_load((EXPERIMENT_DIR / "config.yaml").read_text())
print(json.dumps({
    "status": config["experiment"]["status"],
    "parent": config["lineage"]["parent"],
    "comparison": config["lineage"]["comparison"],
    "validation": config["validation"],
    "model": config["model"],
}, indent=2, ensure_ascii=False))

assert config["data"]["exp264_stage_d_surface"] == (
    "corrected_stage_d_v3_selector_compact_addonly_lgb_mean"
)
assert config["data"]["exp264_stage_c_surface"] == (
    "corrected_stage_c_v6_strict_nested_outer_valid"
)
assert config["model"]["params"]["trained_boosters"] == 0

# %% [markdown]
# ## 3. Input checks

# %%
required_inputs = {
    "exp274_oof": SOURCE_DIR
    / "exp274_catboost_final_regressor_swap_on_exp238_oof_predictions.csv.gz",
    "exp274_by_well": SOURCE_DIR
    / "exp274_catboost_final_regressor_swap_on_exp238_by_well.csv",
    "exp264_stage_c_candidate_score": SOURCE_DIR
    / "exp264_stage_c_v6/artifacts/nested_outer_valid_candidate_score.parquet",
    "exp264_selector_manifest": ROOT
    / "experiments/exp264_exp263_candidate_confidence_dual_selector/kaggle/output/"
    "oof_selector_confidence_probe_v3/artifacts/"
    "exp264_exp263_candidate_confidence_dual_selector_"
    "oof_selector_confidence_probe_plot_manifest.csv",
    "exp264_selector_summary": ROOT
    / "experiments/exp264_exp263_candidate_confidence_dual_selector/kaggle/output/"
    "oof_selector_confidence_probe_v3/artifacts/"
    "exp264_exp263_candidate_confidence_dual_selector_"
    "oof_selector_confidence_probe_summary.json",
}
input_check = pd.DataFrame([
    {"name": name, "exists": path.exists(), "bytes": path.stat().st_size if path.exists() else None, "path": str(path)}
    for name, path in required_inputs.items()
])
display(input_check)
if not input_check["exists"].all():
    raise FileNotFoundError("One or more required saved OOF inputs are missing")

# %% [markdown]
# ## 4. Readout orchestration
#
# `EXP300_RUN_FULL_READOUT=1` のときだけ、次の順で決定的集計を再生成する。
# 主集計は乱数を使わず、補助 logistic AUC のみ `random_state=42` を固定する。

# %%
readout_scripts = [
    "well_feature_readout.py",
    "threshold_readout.py",
    "row_readout.py",
    "selector_readout.py",
    "selector_switch_readout.py",
    "selector_oracle_attribution.py",
]
print("Readout order:", readout_scripts)
if RUN_FULL_READOUT:
    for script_name in readout_scripts:
        command = [str(ROOT / ".venv/bin/python"), str(EXPERIMENT_DIR / script_name)]
        print("Running:", " ".join(command))
        subprocess.run(command, cwd=ROOT, check=True)
else:
    print("Existing verified artifacts are displayed; set EXP300_RUN_FULL_READOUT=1 to regenerate.")

# %% [markdown]
# ## 5. Well and long-tail results

# %%
summary_lines = (ARTIFACT_DIR / "summary.txt").read_text().splitlines()
print("\n".join(summary_lines))

well = pd.read_csv(ARTIFACT_DIR / "well_comparison_and_features.csv")
top_columns = [
    "well", "rows", "exp274_rmse", "exp264_rmse",
    "delta_exp264_vs_exp274", "tail_gr_missing_frac",
    "oracle_tail_tvt_range",
]
display(well.sort_values("delta_exp264_vs_exp274", ascending=False)[top_columns].head(20))

# %%
distance = pd.read_csv(ARTIFACT_DIR / "row_metric_distance_bucket.csv")
tail_decile = pd.read_csv(ARTIFACT_DIR / "row_metric_relative_tail_decile.csv")
gr_missing = pd.read_csv(ARTIFACT_DIR / "row_metric_gr_missing.csv")
fold_match = pd.read_csv(ARTIFACT_DIR / "fold_assignment_summary.csv")
display(distance)
display(tail_decile)
display(gr_missing)
display(fold_match)

# %% [markdown]
# ## 6. Selector results
#
# `dominant_primary_candidate` は各 well 内で最も多かった Stage C hard top1候補であり、
# Stage D v3の最終予測そのものではない。Stage Dはcompact selector特徴74列をadd-onlyで使う。

# %%
selector_summary = json.loads((ARTIFACT_DIR / "selector_readout_summary.json").read_text())
assert selector_summary["source_contract"]["hard_primary_is_final_prediction"] is False
print(json.dumps(selector_summary["selector_findings"], indent=2, ensure_ascii=False))

global_distribution = pd.read_csv(
    ARTIFACT_DIR / "selector_global_candidate_distribution.csv"
)
dominant = pd.read_csv(ARTIFACT_DIR / "selector_dominant_candidate_summary.csv")
effects = pd.read_csv(ARTIFACT_DIR / "selector_metric_effects.csv")
lift = pd.read_csv(ARTIFACT_DIR / "selector_candidate_lift_by_threshold.csv")
display(global_distribution)
display(dominant)
display(effects[effects["threshold"].eq(3.0)])
display(lift[lift["threshold"].eq(3.0)])

# %% [markdown]
# ### Oracle candidate ranking versus Stage D attribution
#
# primary 11候補のactual-error oracle top1を診断上の上限として、
# selector ranking regretとStage D downstream effectを分離する。
# oracleはactual TVTを使うためdeployable routingには使わない。

# %%
oracle_summary = json.loads(
    (ARTIFACT_DIR / "selector_oracle_attribution_summary.json").read_text()
)
oracle_scope = pd.read_csv(
    ARTIFACT_DIR / "selector_oracle_scope_summary.csv"
)
oracle_distance = pd.read_csv(
    ARTIFACT_DIR / "selector_oracle_distance_summary.csv"
)
oracle_wells = pd.read_csv(
    ARTIFACT_DIR / "selector_oracle_top100_worse_wells.csv"
)
display(oracle_scope)
print(json.dumps(oracle_summary["focused_misselection"], indent=2))
display(
    oracle_distance[
        oracle_distance["scope"].eq("worse_gt3")
    ][
        [
            "distance_bucket",
            "selector_tie_aware_correct_rate",
            "selection_abs_regret_mean",
            "oracle_primary_rmse",
            "selected_hard_rmse",
            "stage_d_final_rmse",
            "exp274_rmse",
        ]
    ]
)
display(
    oracle_wells[
        [
            "well",
            "delta_exp264_vs_exp274",
            "selector_tie_aware_correct_rate",
            "oracle_primary_rmse",
            "selected_hard_rmse",
            "stage_d_final_rmse",
            "exp274_rmse",
            "selection_regret_mse",
            "stage_d_vs_selected_mse",
        ]
    ].head(20)
)

# %% [markdown]
# ### Candidate switch attribution
#
# exp274にはselectorがないため、exp264 Stage C hard top1のwell内切替と
# Stage D finalのexp274比悪化が重なるかを読む。previous-candidate holdは
# actual TVTを使うoracle hard-path診断であり、Stage Dの因果ablationではない。

# %%
switch_summary = json.loads(
    (ARTIFACT_DIR / "selector_switch_readout_summary.json").read_text()
)
switch_windows = pd.read_csv(
    ARTIFACT_DIR / "selector_switch_window_summary.csv"
)
run_categories = pd.read_csv(
    ARTIFACT_DIR / "selector_switch_run_category_summary.csv"
)
display(
    switch_windows[
        switch_windows["scope"].isin(["all", "worse_gt3"])
        & switch_windows["window_rows"].isin([0, 5])
    ]
)
display(run_categories[run_categories["scope"].isin(["all", "worse_gt3"])])
print(json.dumps(switch_summary["worse_gt3_vs_other"], indent=2))

# %% [markdown]
# ## 7. Reproducibility and non-use contract

# %%
print("Input SHA256")
print(json.dumps(selector_summary["input_sha256"], indent=2))
print("\nNon-use contract")
for item in selector_summary["non_use_contract"]:
    print("-", item)
for item in switch_summary["non_use_contract"]:
    print("-", item)
for item in oracle_summary["non_use_contract"]:
    print("-", item)

assert selector_summary["rows"] == 3_783_989
assert selector_summary["wells"] == 773
assert selector_summary["status"] == "complete_diagnostic_only"
assert switch_summary["status"] == "complete_diagnostic_only"
assert switch_summary["definition"]["exp274_has_selector"] is False
assert switch_summary["definition"]["hard_top1_is_stage_d_final"] is False
assert oracle_summary["status"] == "complete_diagnostic_only"
assert oracle_summary["definition"]["oracle_is_deployable"] is False
assert oracle_summary["definition"]["hard_top1_is_stage_d_final"] is False
print("Diagnostic readout contract: PASS")
