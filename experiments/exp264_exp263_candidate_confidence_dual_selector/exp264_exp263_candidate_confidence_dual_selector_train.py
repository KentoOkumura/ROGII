# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # exp264 exp263 candidate confidence dual selector — train
#
# exp263 Stage 0を候補bankの正とし、6 primitive + 5 fixed pair + 1 fixed formulaを
# candidate-longへ変換する。Stage Aは特徴量・confidence・重複・相関を監査し、Stage Bは
# `pred_abs_error`と`p_within10`をouter well 5-foldで学習する。Stage Cはouter-train内の
# inner 4-fold OOFとouter-valid向け4-model ensembleから、後段fold別compact metaを作る。
# Stage DはStage Cの74 compact特徴を監査済みexp218 clean 273特徴へadd-onlyし、同一fold・同一GPU設定の
# matched controlと2 variants × 3 configs × 5 folds = 30 boostersで比較する。
# target-derived exp263 catalog/readout/eligibilityは特徴量へjoinしない。

# %% [markdown]
# ## Contents
#
# 1. Imports and runtime helpers
# 2. Stage and compute contract
# 3. Candidate/cache/input contract
# 4. Stage A feature and confidence audit
# 5. Stage B/C selector or Stage D downstream TVT execution
# 6. Metrics and feature importance
# 7. Generated artifacts and reproducibility evidence

# %% [markdown]
# ## 1. Imports and runtime helpers

# %%
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from settings import EXPERIMENT_NAME, ExperimentPaths, load_config

from src.candidate_selector_pipeline import (
    audit_raw_context_availability,
    read_yaml,
    resolve_existing_path,
    resolve_exp263_cache_root,
    resolve_stage_c_artifact_root,
    run_stage_a,
    run_stage_b,
    run_stage_c,
    run_stage_d,
    sha256_file,
    stage_d_cost_contract,
    verify_exp263_root,
)

paths = ExperimentPaths()
paths.require_kaggle_runtime()
paths.ensure_output_dirs()
config = load_config()
output_dir = paths.artifacts_dir


def resolve_support_file(filename: str) -> Path:
    candidates = [
        Path.cwd() / filename,
        paths.experiment_dir / filename,
        Path("/kaggle/working") / filename,
    ]
    matches = [path for path in candidates if path.exists()]
    if not matches:
        matches = list(Path("/kaggle/working").rglob(filename))
    if not matches:
        raise FileNotFoundError(filename)
    return sorted(matches)[0]


# %% [markdown]
# ## 2. Stage and compute contract
#
# `feature_contract_audit`は0 booster。`selector_outer_oof`は1 variant × 2 objectives ×
# 5 outer folds = 10 CPU boosters。`nested_compact_meta`は5 outer × 4 inner × 2 objectives =
# 40 CPU boosters。`downstream_tvt_ablation`はmatched controlとcompact add-onlyを各15本、
# 合計30 GPU boostersで学習する。学習stageは`run_approved=true`のときだけ実行する。

# %%
stage = str(config["execution"]["stage"])
run_approved = bool(config["execution"]["run_approved"])
allowed = set(config["execution"]["allowed_stages"])
if stage not in allowed:
    raise ValueError(f"unknown execution stage: {stage}")
implemented_stages = {
    "feature_contract_audit",
    "selector_outer_oof",
    "nested_compact_meta",
    "downstream_tvt_ablation",
}
if stage not in implemented_stages:
    raise RuntimeError(f"Stage {stage} is outside the implemented Stage A/B/C/D train notebook")
if stage != "feature_contract_audit" and not run_approved:
    raise RuntimeError(f"Stage {stage} requires execution.run_approved=true after cost approval")

stage_cost = {
    "feature_contract_audit": {"variants": 0, "objectives": 0, "folds": 0, "boosters": 0},
    "selector_outer_oof": {"variants": 1, "objectives": 2, "folds": 5, "boosters": 10},
    "nested_compact_meta": {
        "variants": 1,
        "objectives": 2,
        "folds": "5 outer x 4 inner",
        "boosters": 40,
    },
    "downstream_tvt_ablation": {
        "variants": 2,
        "objectives": 3,
        "folds": 5,
        "boosters": 30,
    },
}[stage]

cost_contract = {
    "experiment": EXPERIMENT_NAME,
    "route": config["experiment"]["route"],
    "stage": stage,
    "active_variants": stage_cost["variants"],
    "objectives_or_configs": stage_cost["objectives"],
    "folds": stage_cost["folds"],
    "total_boosters": stage_cost["boosters"],
    "runtime": "kaggle_gpu" if stage == "downstream_tvt_ablation" else "kaggle_cpu",
    "gpu": config["runtime"]["kaggle"]["enable_gpu"],
    "internet": config["runtime"]["kaggle"]["enable_internet"],
    "parent_or_control_retraining": stage == "downstream_tvt_ablation",
}
display(cost_contract)
assert config["experiment"]["route"] == "ml_model"
assert config["model"]["selector_only_stage"]["planned_cpu_boosters"] == 10
assert config["model"]["nested_downstream_stage"]["planned_cpu_selector_boosters"] == 40
if stage == "nested_compact_meta":
    assert config["model"]["nested_downstream_stage"]["enabled"] is True
if stage == "downstream_tvt_ablation":
    approved_cost = stage_d_cost_contract(config)
    assert approved_cost["total_gpu_boosters"] == 30
    assert config["model"]["downstream_tvt_stage"]["enabled"] is True
    assert cost_contract["gpu"] is True
else:
    assert cost_contract["gpu"] is False
assert cost_contract["internet"] is False

print("Leakage contract")
for rule in config["validation"]["leakage_policy"]:
    print("-", rule)

# %% [markdown]
# ## 3. Input contract
#
# Stage A/B/Cではexp263 manifest/catalog SHAを固定する。Stage Dでは完了済みStage Cの
# manifest/schema/25 compact partitionとexp218 source/configを固定し、exp263 Stage Aを再実行しない。

# %%
search_roots = [Path("/kaggle/input"), Path("/tmp"), paths.root]
assert paths.train_data_dir.exists()
candidate_contract = None
cache_root = None
parent_schema_path = None
stage_c_root = None
exp218_source_path = None
exp218_config_path = None
base_feature_allowlist_path = None
hidden_like_assignment_path = None
if stage == "downstream_tvt_ablation":
    stage_c_root = resolve_stage_c_artifact_root(config, search_roots)
    exp218_source_path = resolve_existing_path(
        [str(item) for item in config["data"]["exp218_source_patterns"]], search_roots
    )
    exp218_config_path = resolve_existing_path(
        [str(item) for item in config["data"]["exp218_config_patterns"]], search_roots
    )
    base_feature_allowlist_path = resolve_existing_path(
        [str(item) for item in config["data"]["exp218_clean_273_allowlist_patterns"]],
        search_roots,
    )
    hidden_like_assignment_path = resolve_existing_path(
        [str(item) for item in config["data"]["hidden_like_assignment_patterns"]],
        search_roots,
    )
    input_contract = {
        "stage_c_artifact_root": str(stage_c_root),
        "stage_c_nested_compact_manifest_sha256": sha256_file(
            stage_c_root / "nested_compact_manifest.json"
        ),
        "exp218_source": str(exp218_source_path),
        "exp218_source_sha256": sha256_file(exp218_source_path),
        "exp218_config": str(exp218_config_path),
        "exp218_config_sha256": sha256_file(exp218_config_path),
        "exp218_clean_273_allowlist": str(base_feature_allowlist_path),
        "exp218_clean_273_allowlist_sha256": sha256_file(base_feature_allowlist_path),
        "hidden_like_assignment": str(hidden_like_assignment_path),
        "hidden_like_assignment_sha256": sha256_file(hidden_like_assignment_path),
        "raw_train_dir": str(paths.train_data_dir),
    }
else:
    candidate_contract_path = resolve_support_file("candidate_contract.yaml")
    candidate_contract = read_yaml(candidate_contract_path)
    candidate_names = [str(item["id"]) for item in candidate_contract["score_candidates"]]
    assert len(candidate_names) == 12
    assert len(set(candidate_names)) == 12
    assert candidate_contract["candidate_id_model_encoding"]["type"] == "one_hot"
    assert (
        candidate_contract["candidate_id_model_encoding"]["ordinal_index_as_model_feature"]
        is False
    )
    cache_root = resolve_exp263_cache_root(config, search_roots)
    cache_evidence = verify_exp263_root(cache_root, config)
    parent_schema_path = resolve_existing_path(
        [str(item) for item in config["data"]["exp251_selected_feature_schema_patterns"]],
        search_roots,
    )
    input_contract = {
        "exp263_cache": cache_evidence,
        "candidate_count": len(candidate_names),
        "candidates": candidate_names,
        "primary_legal_domain": candidate_contract["legal_domains"][
            "primitive_pair_bank"
        ]["candidates"],
        "fixed_legal_domain": candidate_contract["legal_domains"]["primitive_fixed_bank"][
            "candidates"
        ],
        "raw_train_dir": str(paths.train_data_dir),
        "exp251_v4_schema": str(parent_schema_path),
        "exp251_v4_schema_sha256": sha256_file(parent_schema_path),
    }
    assert cache_evidence["rows"] == config["guards"]["technical"]["expected_rows"]
    assert cache_evidence["wells"] == config["guards"]["technical"]["expected_wells"]
    raw_context_availability = audit_raw_context_availability(
        paths.train_data_dir,
        paths.test_data_dir,
        config["features"]["raw_context"]["horizontal_numeric_allowlist"],
    )
    raw_context_availability.to_csv(
        output_dir / "raw_context_availability_audit.csv", index=False
    )
    display(raw_context_availability)
display(input_contract)

# %% [markdown]
# ## 4. Stage A feature and confidence audit
#
# `ctx__/cand__/conf__/bank__/formula__/id__`を作り、全欠損・constant・exact duplicateだけを
# deterministicに除去する。|Pearson|または|Spearman|が0.999以上の組は報告のみで落とさない。
# feature schemaはLightGBM fit前にSHA固定する。

# %%
if stage != "downstream_tvt_ablation":
    stage_a_summary = run_stage_a(
        config=config,
        contract=candidate_contract,
        cache_root=cache_root,
        raw_train_dir=paths.train_data_dir,
        output_dir=output_dir,
        parent_schema_path=parent_schema_path,
    )
    display(stage_a_summary)

    feature_catalog = pd.read_csv(output_dir / "feature_catalog.csv")
    correlation_audit = pd.read_csv(output_dir / "feature_duplicate_correlation_audit.csv")
    confidence_coverage = pd.read_csv(output_dir / "confidence_coverage_by_candidate_fold.csv")
    parent_mapping = pd.read_csv(output_dir / "exp251_v4_feature_mapping.csv")
    display(
        feature_catalog.groupby(["group", "selected"], dropna=False).size().rename("features")
    )
    display(correlation_audit.head(100))
    display(
        confidence_coverage.groupby(["candidate_id", "field"], as_index=False)[
            "coverage"
        ].mean()
    )
    display(
        parent_mapping.groupby("action")
        .size()
        .sort_values(ascending=False)
        .rename("exp251_features")
    )
else:
    print("Stage D reuses the frozen Stage C feature/schema audit; Stage A is not rerun.")

# %% [markdown]
# ## 5. Stage B/C selector or Stage D downstream TVT execution
#
# Stage Bは同じrunで固定したallowlistだけを使う。outer-valid wellのlabelはfit、calibration、
# threshold選択へ使わず、valid candidate-long全行へ2 scoreを出す。OOF scoreは監査用Parquetへ
# streaming保存し、同じchunk内でcompact metaへ即変換する。Stage Cはouter-trainをwell単位の
# inner 4-foldへ分け、inner OOFをtrain role、4-model ensembleをouter-valid roleとして25個の
# downstream-fold partitionへ保存する。
# Stage Dは25 partitionをfold契約の正として読み、matched control 273列とcompact add-only
# 347列を同一行・同一3-config GPU条件で学習する。hidden-like assignmentは事後評価だけに使う。

# %%
selector_summary = None
if stage == "feature_contract_audit":
    print("Stage A completed: 0 variants / 0 configs / 0 folds / 0 boosters")
elif stage == "selector_outer_oof":
    selector_summary = run_stage_b(
        config=config,
        contract=candidate_contract,
        cache_root=cache_root,
        raw_train_dir=paths.train_data_dir,
        output_dir=output_dir,
    )
    assert selector_summary["model_count"] == 10
    display(selector_summary)
elif stage == "nested_compact_meta":
    selector_summary = run_stage_c(
        config=config,
        contract=candidate_contract,
        cache_root=cache_root,
        raw_train_dir=paths.train_data_dir,
        output_dir=output_dir,
    )
    assert selector_summary["model_count"] == 40
    assert selector_summary["leakage_audit"]["passed"] is True
    display(selector_summary)
else:
    selector_summary = run_stage_d(
        config=config,
        stage_c_root=stage_c_root,
        exp218_source_path=exp218_source_path,
        exp218_config_path=exp218_config_path,
        base_feature_allowlist_path=base_feature_allowlist_path,
        hidden_like_assignment_path=hidden_like_assignment_path,
        raw_train_dir=paths.train_data_dir,
        output_dir=output_dir,
    )
    assert selector_summary["model_count"] == 30
    display(selector_summary)

# %% [markdown]
# ## 6. Metrics and feature importance

# %%
if stage == "selector_outer_oof":
    selector_metrics = pd.read_csv(output_dir / "selector_metrics.csv")
    importance = pd.read_csv(output_dir / "feature_importance_by_objective_fold.csv")
    selection = pd.read_csv(output_dir / "selector_selection_rate.csv")
    by_well = pd.read_csv(output_dir / "selector_by_well.csv")
    display(selector_metrics)
    display(
        selection.groupby("candidate_id", as_index=False)["selected_rows"]
        .sum()
        .sort_values("selected_rows", ascending=False)
    )
    display(by_well.sort_values("hard_primary_rmse", ascending=False).head(80))

    importance_mean = (
        importance.groupby(["objective", "feature"], as_index=False)["gain_importance"]
        .mean()
        .sort_values(["objective", "gain_importance"], ascending=[True, False])
    )
    for objective in ["pred_abs_error", "p_within10"]:
        top = importance_mean[importance_mean["objective"].eq(objective)].head(30)
        display(top)
        if len(top):
            ax = top.sort_values("gain_importance").plot.barh(
                x="feature",
                y="gain_importance",
                figsize=(10, 10),
                legend=False,
                title=f"exp264 {objective} mean gain importance",
            )
            ax.set_xlabel("mean gain importance across outer folds")
            plt.tight_layout()
            plt.savefig(output_dir / f"feature_importance_{objective}_top30.png", dpi=140)
            plt.show()
elif stage == "nested_compact_meta":
    selector_metrics = pd.read_csv(output_dir / "nested_selector_metrics.csv")
    fold_manifest = pd.read_csv(output_dir / "nested_fold_manifest.csv")
    partition_manifest = pd.read_csv(output_dir / "nested_compact_partition_manifest.csv")
    importance = pd.read_csv(
        output_dir / "nested_feature_importance_by_objective_outer_inner.csv"
    )
    display(selector_metrics)
    display(fold_manifest)
    display(
        partition_manifest.groupby(["downstream_outer_fold", "role"], as_index=False)["rows"]
        .sum()
        .sort_values(["downstream_outer_fold", "role"])
    )
    importance_mean = (
        importance[importance["importance_type"].eq("gain")]
        .groupby(["objective", "feature"], as_index=False)["importance"]
        .mean()
        .sort_values(["objective", "importance"], ascending=[True, False])
    )
    for objective in ["pred_abs_error", "p_within10"]:
        top = importance_mean[importance_mean["objective"].eq(objective)].head(30)
        display(top)
        if len(top):
            ax = top.sort_values("importance").plot.barh(
                x="feature",
                y="importance",
                figsize=(10, 10),
                legend=False,
                title=f"exp264 Stage C {objective} mean gain importance",
            )
            ax.set_xlabel("mean gain importance across outer x inner models")
            plt.tight_layout()
            plt.savefig(output_dir / f"nested_feature_importance_{objective}_top30.png", dpi=140)
            plt.show()
elif stage == "downstream_tvt_ablation":
    stage_d_metrics = json.loads((output_dir / "stage_d_metrics.json").read_text())
    fold_metrics = pd.read_csv(output_dir / "stage_d_fold_metrics.csv")
    bucket_metrics = pd.read_csv(output_dir / "stage_d_bucket_metrics.csv")
    hidden_metrics = pd.read_csv(output_dir / "stage_d_hidden_like_metrics.csv")
    by_well = pd.read_csv(output_dir / "stage_d_by_well.csv")
    importance = pd.read_csv(output_dir / "stage_d_feature_importance.csv")
    display(stage_d_metrics)
    display(fold_metrics[fold_metrics["model"].eq("lgb_mean")])
    display(bucket_metrics)
    display(hidden_metrics)
    display(by_well.sort_values("delta_rmse_addonly_minus_control", ascending=False).head(80))
    importance_mean = (
        importance[
            importance["importance_type"].eq("gain")
            & importance["variant"].eq("selector_compact_addonly")
        ]
        .groupby("feature", as_index=False)["importance"]
        .mean()
        .sort_values("importance", ascending=False)
    )
    display(importance_mean.head(100))
    compact_importance = importance_mean[
        importance_mean["feature"].str.startswith("selector__")
    ].head(50)
    display(compact_importance)
    if len(compact_importance):
        ax = compact_importance.head(30).sort_values("importance").plot.barh(
            x="feature",
            y="importance",
            figsize=(10, 11),
            legend=False,
            title="exp264 Stage D compact feature mean gain importance",
        )
        ax.set_xlabel("mean gain importance across 15 add-only models")
        plt.tight_layout()
        plt.savefig(output_dir / "stage_d_compact_feature_importance_top30.png", dpi=140)
        plt.show()

# %% [markdown]
# ## 7. Generated artifacts and reproducibility evidence

# %%
print("Generated files")
for generated in sorted(output_dir.rglob("*")):
    if generated.is_file():
        print(generated.relative_to(output_dir), generated.stat().st_size)

reproducibility = json.loads((output_dir / "reproducibility_manifest.json").read_text())
display(reproducibility)
