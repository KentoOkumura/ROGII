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
# # exp287 fold-safe formation 74 add-only on exp264 — train
#
# 修正版exp264 Stage C v6 / Stage D v3を固定し、exp218監査で
# `full_train_formation_reference`依存と判定されたbase-replay 74列だけをouter fold内で
# 再生成する。clean 273 + nested compact 74 + fold-safe formation 74 = 421列の1 variantを、
# 3 LightGBM config × 5 folds = 15 GPU boostersで評価する。保存済みexp264 347列OOFを
# controlに使い、control boosterは再学習しない。

# %% [markdown]
# ## Contents
#
# 1. Imports and runtime helpers
# 2. Execution and GPU cost contract
# 3. Frozen parent and feature contracts
# 4. Fold-safe formation generation contract
# 5. Preflight or 15-booster training orchestration
# 6. Metrics, guards, and feature importance
# 7. Generated artifacts and reproducibility evidence

# %% [markdown]
# ## 1. Imports and runtime helpers

# %%
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import yaml
from IPython.display import display

from src.candidate_selector_pipeline import (
    resolve_existing_path,
    resolve_stage_c_artifact_root,
    sha256_file,
)
from src.fold_safe_formation_pipeline import (
    canonical_formation_feature_names,
    formation_cost_contract,
    load_formation_feature_contract,
    run_fold_safe_formation_train,
    run_preflight,
)

EXPERIMENT_NAME = "exp287_fold_safe_formation_74_addonly_on_exp264"
PACKAGE_DIR = Path.cwd()
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")


def read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text()) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def find_config_path() -> Path:
    candidates = [
        PACKAGE_DIR / "config.yaml",
        PACKAGE_DIR / "experiments" / EXPERIMENT_NAME / "config.yaml",
    ]
    candidates.extend(PACKAGE_DIR.rglob(f"{EXPERIMENT_NAME}/config.yaml"))
    matches = sorted({path.resolve() for path in candidates if path.exists()})
    if len(matches) != 1:
        raise FileNotFoundError(f"exp287 config resolution is ambiguous: {matches}")
    return matches[0]


def find_competition_input_root() -> Path:
    preferred = [
        KAGGLE_INPUT_ROOT
        / "competitions"
        / "rogii-wellbore-geology-prediction",
        KAGGLE_INPUT_ROOT / "rogii-wellbore-geology-prediction",
    ]
    candidates = [
        path.resolve()
        for path in preferred
        if path.is_dir() and (path / "train").is_dir() and (path / "test").is_dir()
    ]
    if not candidates and KAGGLE_INPUT_ROOT.exists():
        candidates = [
            path.resolve()
            for path in KAGGLE_INPUT_ROOT.glob("*/*")
            if path.is_dir() and (path / "train").is_dir() and (path / "test").is_dir()
        ]
    candidates = sorted(set(candidates))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"competition input with train/test directories was not unique: {candidates}"
        )
    return candidates[0]


if not KAGGLE_INPUT_ROOT.exists() or not KAGGLE_WORKING_ROOT.exists():
    raise RuntimeError("Kaggle Notebook execution is authoritative for exp287")
config = read_yaml(find_config_path())
competition_input_root = find_competition_input_root()
raw_train_dir = competition_input_root / "train"
raw_test_dir = competition_input_root / "test"
output_dir = KAGGLE_WORKING_ROOT / "artifacts"
output_dir.mkdir(parents=True, exist_ok=True)


# %% [markdown]
# ## 2. Execution and GPU cost contract
#
# `preflight_only`は0 boosterで、入力SHA、raw train/current-test header、固定74列、
# 保存済みexp264 OOFを確認する。`fold_safe_formation_addonly_train`はfeature cacheを5 fold分
# 生成し、全duplicate/correlation監査を完了してから15 boostersだけを学習する。

# %%
stage = str(config["execution"]["stage"])
allowed_stages = set(config["execution"]["allowed_stages"])
if stage not in allowed_stages:
    raise ValueError(f"unknown execution stage: {stage}")
run_approved = bool(config["execution"]["run_approved"])
if stage == "fold_safe_formation_addonly_train" and not run_approved:
    raise RuntimeError(
        "15-booster Kaggle train requires explicit approval and execution.run_approved=true"
    )

cost_contract = formation_cost_contract(config)
execution_contract = {
    "experiment": EXPERIMENT_NAME,
    "route": config["experiment"]["route"],
    "stage": stage,
    "active_variants": 0 if stage == "preflight_only" else 1,
    "lightgbm_configs": 0 if stage == "preflight_only" else 3,
    "folds": 0 if stage == "preflight_only" else 5,
    "total_gpu_boosters": 0 if stage == "preflight_only" else 15,
    "parent_control_retraining": False,
    "inference": False,
    "submission": False,
}
display(execution_contract)
display(cost_contract)
assert config["experiment"]["route"] == "ml_model"
assert cost_contract["planned_gpu_boosters"] == 15
assert cost_contract["parent_control_retraining"] is False
assert config["runtime"]["kaggle"]["enable_internet"] is False
assert config["runtime"]["kaggle"]["enable_gpu"] is True


# %% [markdown]
# ## 3. Frozen parent and feature contracts
#
# Stage C v6の25 compact partitions、clean-273 allowlist、formation availability audit、
# corrected Stage D v3 OOF、hidden-like assignmentをSHA固定する。旧formation列はclean 273を
# 抽出した直後に破棄され、保存済み347列controlは比較だけに使う。

# %%
search_roots = [KAGGLE_INPUT_ROOT, Path("/tmp"), PACKAGE_DIR]
stage_c_root = resolve_stage_c_artifact_root(config, search_roots)
exp218_source_path = resolve_existing_path(
    [str(value) for value in config["data"]["exp218_source_patterns"]], search_roots
)
exp218_config_path = resolve_existing_path(
    [str(value) for value in config["data"]["exp218_config_patterns"]], search_roots
)
clean_allowlist_path = resolve_existing_path(
    [str(value) for value in config["data"]["clean_273_allowlist_patterns"]], search_roots
)
availability_audit_path = resolve_existing_path(
    [str(value) for value in config["data"]["formation_availability_audit_patterns"]],
    search_roots,
)
saved_parent_oof_path = resolve_existing_path(
    [str(value) for value in config["data"]["saved_exp264_stage_d_oof_patterns"]],
    search_roots,
)
hidden_like_assignment_path = resolve_existing_path(
    [str(value) for value in config["data"]["hidden_like_assignment_patterns"]],
    search_roots,
)

formation_features, formation_contract = load_formation_feature_contract(
    availability_audit_path,
    expected_sha256=config["data"]["formation_availability_audit_sha256"],
)
assert formation_features == canonical_formation_feature_names()
assert len(formation_features) == 74
input_contract = {
    "stage_c_root": str(stage_c_root),
    "exp218_source": str(exp218_source_path),
    "exp218_source_sha256": sha256_file(exp218_source_path),
    "exp218_config": str(exp218_config_path),
    "exp218_config_sha256": sha256_file(exp218_config_path),
    "clean_273_allowlist": str(clean_allowlist_path),
    "clean_273_allowlist_sha256": sha256_file(clean_allowlist_path),
    "formation_availability_audit": formation_contract,
    "saved_exp264_corrected_stage_d_oof": str(saved_parent_oof_path),
    "saved_exp264_corrected_stage_d_oof_sha256": sha256_file(saved_parent_oof_path),
    "hidden_like_assignment": str(hidden_like_assignment_path),
    "hidden_like_assignment_sha256": sha256_file(hidden_like_assignment_path),
    "raw_train_dir": str(raw_train_dir),
    "raw_current_test_dir": str(raw_test_dir),
}
display(input_contract)
display(pd.DataFrame({"position": range(74), "feature": formation_features}))


# %% [markdown]
# ## 4. Fold-safe formation generation contract
#
# - outer-train target: outer-train referenceをfitし、対象well自身のplane/dense sampleを除外する。
# - outer-valid target: outer-train referenceだけを使う。
# - current-test contract: all-train referenceを使い、target horizontalからは
#   `MD/X/Y/Z/TVT_input`だけを読む。formation列は読まない。
# - 74列は欠損・nonfinite、既存347列とのexact duplicate、固定sampleのPearson/Spearmanを
#   fit前に監査する。相関によるpruneは行わない。

# %%
generation_contract = {
    **config["formation_generator"],
    "train_feature_surface": "clean273+nested_compact74+fold_safe_formation74=421",
    "outer_train_reference": "outer_train_wells_with_target_well_self_exclusion",
    "outer_valid_reference": "outer_train_wells_only",
    "current_test_reference": "all_train_wells",
    "current_test_target_horizontal_columns": ["MD", "X", "Y", "Z", "TVT_input"],
    "current_test_target_formation_columns_read": False,
    "correlation_pruning": False,
}
display(generation_contract)
print("Leakage policy")
for rule in config["validation"]["leakage_policy"]:
    print("-", rule)
print("Forbidden actions")
for rule in config["guards"]["forbidden"]:
    print("-", rule)


# %% [markdown]
# ## 5. Preflight or 15-booster training orchestration

# %%
if stage == "preflight_only":
    run_summary = run_preflight(
        config=config,
        stage_c_root=stage_c_root,
        exp218_source_path=exp218_source_path,
        exp218_config_path=exp218_config_path,
        clean_allowlist_path=clean_allowlist_path,
        availability_audit_path=availability_audit_path,
        saved_parent_oof_path=saved_parent_oof_path,
        raw_train_dir=raw_train_dir,
        raw_test_dir=raw_test_dir,
        hidden_like_assignment_path=hidden_like_assignment_path,
        output_dir=output_dir,
        verify_stage_c_partition_sha256=False,
    )
else:
    run_summary = run_fold_safe_formation_train(
        config=config,
        stage_c_root=stage_c_root,
        exp218_source_path=exp218_source_path,
        exp218_config_path=exp218_config_path,
        clean_allowlist_path=clean_allowlist_path,
        availability_audit_path=availability_audit_path,
        saved_parent_oof_path=saved_parent_oof_path,
        raw_train_dir=raw_train_dir,
        raw_test_dir=raw_test_dir,
        hidden_like_assignment_path=hidden_like_assignment_path,
        output_dir=output_dir,
    )
display(run_summary)


# %% [markdown]
# ## 6. Metrics, guards, and feature importance

# %%
if stage == "preflight_only":
    schema_audit = pd.read_csv(output_dir / "raw_train_current_test_schema_audit.csv")
    display(
        schema_audit.groupby("split", as_index=False).agg(
            wells=("well", "nunique"), passed=("passed", "all")
        )
    )
    print("Preflight completed with 0 boosters; no prediction or submission was generated.")
else:
    metrics = json.loads((output_dir / "metrics.json").read_text())
    fold_metrics = pd.read_csv(output_dir / "fold_metrics.csv")
    bucket_metrics = pd.read_csv(output_dir / "bucket_metrics.csv")
    hidden_metrics = pd.read_csv(output_dir / "hidden_like_metrics.csv")
    by_well = pd.read_csv(output_dir / "by_well_metrics.csv")
    importance = pd.read_csv(output_dir / "feature_importance.csv")
    relationships = pd.read_csv(output_dir / "formation_feature_relationship_audit.csv")
    display(metrics["guard"])
    display(fold_metrics[fold_metrics["model"].eq("lgb_mean")])
    display(bucket_metrics)
    display(hidden_metrics)
    display(by_well.sort_values("new_minus_parent_delta", ascending=False).head(80))
    display(
        relationships.sort_values(
            ["exact_duplicate_count", "max_abs_pearson"], ascending=[False, False]
        ).head(100)
    )
    formation_importance = (
        importance[
            importance["importance_type"].eq("gain")
            & importance["feature_group"].eq("fold_safe_formation")
        ]
        .groupby("feature", as_index=False)["importance"]
        .mean()
        .sort_values("importance", ascending=False)
    )
    display(formation_importance.head(74))
    if len(formation_importance):
        axis = (
            formation_importance.head(30)
            .sort_values("importance")
            .plot.barh(
                x="feature",
                y="importance",
                figsize=(10, 11),
                legend=False,
                title="exp287 fold-safe formation mean gain importance",
            )
        )
        axis.set_xlabel("mean gain importance across 15 models")
        plt.tight_layout()
        plt.savefig(output_dir / "fold_safe_formation_feature_importance_top30.png", dpi=140)
        plt.show()


# %% [markdown]
# ## 7. Generated artifacts and reproducibility evidence

# %%
print("Generated files")
for generated in sorted(output_dir.rglob("*")):
    if generated.is_file():
        print(generated.relative_to(output_dir), generated.stat().st_size)

if stage == "fold_safe_formation_addonly_train":
    reproducibility = json.loads((output_dir / "reproducibility_manifest.json").read_text())
    display(reproducibility)
    assert reproducibility["cost_contract"]["planned_gpu_boosters"] == 15
    assert reproducibility["cost_contract"]["parent_control_retraining"] is False
else:
    preflight_manifest = json.loads((output_dir / "preflight_manifest.json").read_text())
    display(preflight_manifest)
