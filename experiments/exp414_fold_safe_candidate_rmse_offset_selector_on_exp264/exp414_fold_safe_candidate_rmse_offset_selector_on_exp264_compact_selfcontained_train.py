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
# # exp414 fold-safe candidate-RMSE offset selector on exp264 — train
#
# exp407の悪化を、保存済みparent/exp407 candidate-score OOFの定数shiftと
# row-local変化へ分解する。固定treatmentは候補別RMSEをsample weightにせず、
# `pred_abs_error`回帰のadditive base offsetとして使う。
#
# 候補、88列feature、fold、sample row IDs、LightGBM設定はcorrected exp264 Stage B v5
# と同一。新規学習は1 variant × 1 objective × 5 outer folds = 5 CPU boostersだけで、
# classifier、control再学習、GPU、Stage C、inference、submissionは0。

# %% [markdown]
# ## Contents
#
# 1. Imports and immutable boundary
# 2. Runtime, paths, and pinned input resolution
# 3. Candidate, leakage, method, and compute contract
# 4. Input SHA and raw-context checks
# 5. Parent-compatible Stage A feature freeze
# 6. Fold-safe RMSE-offset Stage B
# 7. Root-cause counterfactual and parent gate
# 8. Metrics, diagnostics, and generated artifacts

# %% [markdown]
# ## 1. Imports and immutable boundary

# %%
from __future__ import annotations

import glob
import json
import os
import time
from pathlib import Path
from typing import Sequence

import pandas as pd

from src.candidate_rmse_offset_selector import (
    EXPERIMENT_NAME,
    evaluate_rmse_offset_gate,
    run_rmse_offset_stage_b,
    validate_static_contract,
)
from src.candidate_rmse_root_cause_readout import run_root_cause_readout
from src.candidate_selector_pipeline import (
    audit_raw_context_availability,
    read_yaml,
    resolve_exp263_cache_root,
    run_stage_a,
    sha256_file,
)


KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")
APPROVED_SCOPE = (
    "rmse_offset_stage_b_1_variant_1_objective_5_outer_5_cpu_boosters_"
    "no_control_no_classifier_no_gpu_no_inference_no_submission"
)


def in_notebook_runtime() -> bool:
    try:
        return get_ipython() is not None  # type: ignore[name-defined]
    except NameError:
        return False


EXECUTE_NOTEBOOK = (
    os.environ.get("EXP414_IMPORT_ONLY", "0") != "1" and in_notebook_runtime()
)
if EXECUTE_NOTEBOOK:
    import matplotlib.pyplot as plt
    from IPython.display import display

# %% [markdown]
# ## 2. Runtime, paths, and pinned input resolution
#
# Kaggle bootstrap後のcwd、repo root、Kaggle inputsを同じresolverで扱う。
# SHAが固定値と一致するfileだけを採用し、同名fileの偶然の取り違えを防ぐ。

# %%
def project_root() -> Path:
    start = Path.cwd()
    for candidate in (start, *start.parents):
        if (candidate / "project.yml").exists():
            return candidate
    return start


def experiment_dir() -> Path:
    candidate = project_root() / "experiments" / EXPERIMENT_NAME
    return candidate if candidate.exists() else Path.cwd()


def find_support_file(filename: str) -> Path:
    candidates = [
        Path.cwd() / filename,
        experiment_dir() / filename,
        KAGGLE_WORKING_ROOT / filename,
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"support file not found: {filename}")


def runtime_output_dir() -> Path:
    if EXECUTE_NOTEBOOK and KAGGLE_WORKING_ROOT.exists():
        return KAGGLE_WORKING_ROOT / "artifacts"
    return experiment_dir() / "artifacts"


def search_roots() -> list[Path]:
    return [
        Path.cwd(),
        project_root(),
        KAGGLE_INPUT_ROOT,
        Path("/tmp/kaggle-output"),
    ]


def resolve_pinned_file(
    patterns: Sequence[str],
    expected_sha256: str,
    *,
    label: str,
) -> Path:
    matches: set[Path] = set()
    for raw_pattern in patterns:
        pattern = str(raw_pattern)
        direct = Path(pattern)
        if direct.is_file():
            matches.add(direct)
        if direct.is_absolute():
            matches.update(
                Path(item)
                for item in glob.glob(pattern, recursive=True)
                if Path(item).is_file()
            )
            continue
        for root in search_roots():
            if root.exists():
                matches.update(
                    path for path in root.glob(pattern) if path.is_file()
                )
    valid = sorted(
        path for path in matches if sha256_file(path) == str(expected_sha256)
    )
    if not valid:
        observed = {str(path): sha256_file(path) for path in sorted(matches)}
        raise FileNotFoundError(
            f"{label} did not resolve with pinned SHA {expected_sha256}; "
            f"observed={observed}"
        )
    return valid[0]


def find_raw_split(split: str) -> Path:
    candidates = [
        project_root() / "data" / "raw" / split,
        Path.cwd() / "data" / "raw" / split,
        KAGGLE_INPUT_ROOT
        / "competitions"
        / "rogii-wellbore-geology-prediction"
        / split,
        KAGGLE_INPUT_ROOT / "rogii-wellbore-geology-prediction" / split,
    ]
    for path in candidates:
        if path.is_dir() and any(path.glob("*__horizontal_well.csv")):
            return path
    matches = [
        path
        for path in KAGGLE_INPUT_ROOT.glob(f"**/{split}")
        if path.is_dir() and any(path.glob("*__horizontal_well.csv"))
    ]
    if len(matches) != 1:
        raise FileNotFoundError(f"raw {split} directory did not resolve: {matches}")
    return matches[0]


CONFIG_PATH = find_support_file("config.yaml")
CONFIG = read_yaml(CONFIG_PATH)
CONTRACT_PATH = resolve_pinned_file(
    CONFIG["data"]["candidate_contract_patterns"],
    CONFIG["data"]["candidate_contract_file_sha256"],
    label="candidate contract",
)
CONTRACT = read_yaml(CONTRACT_PATH)
OUTPUT_DIR = runtime_output_dir()

# %% [markdown]
# ## 3. Candidate, leakage, method, and compute contract
#
# RMSEは各outer modelのexact sampled fit rowsからだけ計算する。
# 学習targetは`actual_abs_error - fit_candidate_rmse`、最終scoreは
# `max(0, residual_prediction + fit_candidate_rmse)`。
# sample weightは渡さず、binary modelもfitしない。

# %%
STATIC = validate_static_contract(CONFIG, CONTRACT)
assert STATIC["candidate_count"] == 12
assert len(STATIC["primary_domain"]) == 11
assert STATIC["feature_count"] == 88
assert STATIC["cost"] == {
    "active_variants": 1,
    "objectives": 1,
    "outer_folds": 5,
    "planned_cpu_boosters": 5,
    "parent_control_retraining": False,
    "classifier_boosters": 0,
    "gpu_boosters": 0,
    "pf_hmm_beam_regeneration": False,
    "inference": False,
    "submission": False,
}
print(
    json.dumps(
        {
            "experiment": EXPERIMENT_NAME,
            "route": CONFIG["experiment"]["route"],
            "candidate_order": STATIC["candidate_order"],
            "feature_schema_sha256": CONFIG["features"][
                "feature_schema_logical_sha256"
            ],
            "method": CONFIG["candidate_rmse_offset"],
            "cost": STATIC["cost"],
            "run_approved": bool(CONFIG["execution"]["run_approved"]),
            "approval_scope": CONFIG["execution"]["approval_scope"],
        },
        indent=2,
    )
)
print("Leakage policy")
for rule in CONFIG["validation"]["leakage_policy"]:
    print("-", rule)

# %% [markdown]
# ## 4. Input SHA and raw-context checks
#
# exp263 candidate cacheだけを学習入力とする。親corrected exp264とexp407のOOFは
# root-cause counterfactualと比較gateにだけ使い、fit partition offsetへは渡さない。

# %%
if EXECUTE_NOTEBOOK:
    if not bool(CONFIG["execution"]["run_approved"]):
        raise RuntimeError(
            "exp414 Kaggle execution is not approved. Keep "
            "execution.run_approved=false until canonical Notebook adoption "
            "and 1 variant / 1 objective / 5 outer folds / 5 CPU boosters are "
            "separately approved."
        )
    if str(CONFIG["execution"]["approval_scope"]) != APPROVED_SCOPE:
        raise RuntimeError("exp414 approval scope differs from the frozen cost")

    started = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_train_dir = find_raw_split("train")
    raw_test_dir = find_raw_split("test")
    cache_root = resolve_exp263_cache_root(CONFIG, search_roots())
    parent_schema_path = resolve_pinned_file(
        CONFIG["data"]["exp251_selected_feature_schema_patterns"],
        CONFIG["data"]["exp251_selected_feature_schema_sha256"],
        label="exp251 selected feature schema",
    )
    parent_oof_path = resolve_pinned_file(
        CONFIG["data"]["parent_candidate_score_oof"]["patterns"],
        CONFIG["data"]["parent_candidate_score_oof"]["sha256"],
        label="corrected exp264 Stage B v5 candidate-score OOF",
    )
    exp407_oof_path = resolve_pinned_file(
        CONFIG["data"]["exp407_candidate_score_oof"]["patterns"],
        CONFIG["data"]["exp407_candidate_score_oof"]["sha256"],
        label="exp407 Stage B v1 candidate-score OOF",
    )
    parent_cfg = CONFIG["data"]["parent_stage_b"]
    parent_metrics_path = resolve_pinned_file(
        parent_cfg["selector_metrics_patterns"],
        parent_cfg["selector_metrics_sha256"],
        label="parent selector metrics",
    )
    parent_bucket_path = resolve_pinned_file(
        parent_cfg["distance_bucket_patterns"],
        parent_cfg["distance_bucket_sha256"],
        label="parent distance-bucket metrics",
    )
    parent_by_well_path = resolve_pinned_file(
        parent_cfg["by_well_patterns"],
        parent_cfg["by_well_sha256"],
        label="parent by-well metrics",
    )
    hidden_cfg = CONFIG["data"]["hidden_like_assignment"]
    hidden_like_path = resolve_pinned_file(
        hidden_cfg["patterns"],
        hidden_cfg["sha256"],
        label="hidden-like assignment",
    )
    weight_cfg = CONFIG["data"]["exp407_weight_table"]
    exp407_weight_path = resolve_pinned_file(
        weight_cfg["patterns"],
        weight_cfg["sha256"],
        label="exp407 fold-safe inverse-RMSE weight table",
    )
    INPUTS = {
        "config": {"path": str(CONFIG_PATH), "sha256": sha256_file(CONFIG_PATH)},
        "candidate_contract": {
            "path": str(CONTRACT_PATH),
            "sha256": sha256_file(CONTRACT_PATH),
        },
        "raw_train_dir": str(raw_train_dir),
        "raw_test_dir": str(raw_test_dir),
        "exp263_cache_root": str(cache_root),
        "parent_schema": {
            "path": str(parent_schema_path),
            "sha256": sha256_file(parent_schema_path),
        },
        "parent_candidate_score_oof": {
            "path": str(parent_oof_path),
            "sha256": sha256_file(parent_oof_path),
        },
        "exp407_candidate_score_oof": {
            "path": str(exp407_oof_path),
            "sha256": sha256_file(exp407_oof_path),
        },
        "exp407_weight_table": {
            "path": str(exp407_weight_path),
            "sha256": sha256_file(exp407_weight_path),
        },
        "hidden_like_assignment": {
            "path": str(hidden_like_path),
            "sha256": sha256_file(hidden_like_path),
        },
    }
    print(json.dumps(INPUTS, indent=2))

    availability = audit_raw_context_availability(
        raw_train_dir,
        raw_test_dir,
        CONFIG["features"]["raw_context"]["horizontal_numeric_allowlist"],
    )
    availability.to_csv(
        OUTPUT_DIR / "raw_context_availability_audit.csv", index=False
    )
    display(availability)
    if not bool(availability["availability_pass"].all()):
        raise RuntimeError("raw train/test context availability audit failed")

# %% [markdown]
# ## 5. Parent-compatible Stage A feature freeze
#
# 親と同じraw context、candidate shape、bank relation、confidence、
# candidate/family/formula one-hotを再構成する。selected 88列とlogical SHAが
# corrected exp264から1列でも変われば、offset計算とfitへ進まない。

# %%
if EXECUTE_NOTEBOOK:
    stage_a = run_stage_a(
        config=CONFIG,
        contract=CONTRACT,
        cache_root=cache_root,
        raw_train_dir=raw_train_dir,
        output_dir=OUTPUT_DIR,
        parent_schema_path=parent_schema_path,
    )
    if int(stage_a["selected_feature_count"]) != 88:
        raise RuntimeError("exp414 Stage A did not reproduce 88 selected features")
    if stage_a["feature_schema_sha256"] != CONFIG["features"][
        "feature_schema_logical_sha256"
    ]:
        raise RuntimeError("exp414 Stage A schema differs from corrected exp264")
    display(stage_a)
    feature_catalog = pd.read_csv(OUTPUT_DIR / "feature_catalog.csv")
    display(
        feature_catalog.groupby(["group", "selected"], dropna=False)
        .size()
        .rename("features")
    )

# %% [markdown]
# ## 6. Fold-safe RMSE-offset Stage B
#
# 各outer foldについて、親と同じsampled fit row IDsから12候補のRMSE offsetを作り、
# unweighted residual L1 regressorを1本だけ学習する。offset table、fit row-ID SHA、
# residual target SHA、feature content SHA、model SHA、OOF SHAを保存する。

# %%
if EXECUTE_NOTEBOOK:
    stage_b = run_rmse_offset_stage_b(
        config=CONFIG,
        contract=CONTRACT,
        cache_root=cache_root,
        raw_train_dir=raw_train_dir,
        output_dir=OUTPUT_DIR,
    )
    if int(stage_b["model_count"]) != 5:
        raise RuntimeError("exp414 Stage B did not produce exactly 5 regressors")
    if int(stage_b["classifier_model_count"]) != 0:
        raise RuntimeError("exp414 unexpectedly trained a classifier")
    if int(stage_b["candidate_score_oof_rows"]) != int(
        CONFIG["data"]["expected_candidate_long_oof_rows"]
    ):
        raise RuntimeError("exp414 candidate-score OOF row count changed")
    offset_manifest = stage_b["candidate_rmse_offset"]
    if not bool(offset_manifest["all_checks_passed"]):
        raise RuntimeError("exp414 candidate RMSE offset audit failed")
    display(stage_b)
    display(pd.read_csv(OUTPUT_DIR / "candidate_rmse_offset_by_fold.csv"))
    display(
        pd.read_csv(OUTPUT_DIR / "candidate_rmse_offset_truth_read_ledger.csv")
    )

# %% [markdown]
# ## 7. Root-cause counterfactual and parent gate
#
# parentにexp407のcandidate×fold平均shiftだけを適用した`global_shift_only`と、
# exp407から同平均を除いた`local_change_only`を比較する。weightとrow-local driftの
# dose-response、親margin別damageも同じreadoutで再計算する。
#
# treatmentは親と同じ科学gate（score MAE、hard RMSE、fold、near/1000+、
# hidden-like 2面、worst well）へ通し、全ANDのときだけ方法確立と判定する。

# %%
if EXECUTE_NOTEBOOK:
    root_cause = run_root_cause_readout(
        parent_path=parent_oof_path,
        exp407_path=exp407_oof_path,
        treatment_path=OUTPUT_DIR / "candidate_score_oof.parquet",
        exp407_weight_table_path=exp407_weight_path,
        candidate_order=STATIC["candidate_order"],
        primary_domain=STATIC["primary_domain"],
        output_dir=OUTPUT_DIR,
        config=CONFIG,
    )
    if not bool(root_cause["root_cause_gate"]["passed"]):
        raise RuntimeError(
            "root-cause gate did not reproduce the preregistered exp407 diagnosis"
        )
    gate = evaluate_rmse_offset_gate(
        config=CONFIG,
        contract=CONTRACT,
        output_dir=OUTPUT_DIR,
        parent_metrics_path=parent_metrics_path,
        parent_bucket_path=parent_bucket_path,
        parent_by_well_path=parent_by_well_path,
        hidden_like_assignment_path=hidden_like_path,
        root_cause_summary_path=OUTPUT_DIR / "root_cause_summary.json",
        source_config_path=CONFIG_PATH,
        source_candidate_contract_path=CONTRACT_PATH,
    )
    display(root_cause)
    display(gate)
    display(pd.read_csv(OUTPUT_DIR / "exp414_parent_fold_comparison.csv"))
    display(pd.read_csv(OUTPUT_DIR / "exp414_parent_bucket_comparison.csv"))
    display(pd.read_csv(OUTPUT_DIR / "exp414_parent_hidden_like_comparison.csv"))
    print("Decision:", gate["decision"])

# %% [markdown]
# ## 8. Metrics, diagnostics, and generated artifacts

# %%
if EXECUTE_NOTEBOOK:
    metrics = pd.read_csv(OUTPUT_DIR / "selector_metrics.csv")
    candidate_metrics = pd.read_csv(
        OUTPUT_DIR / "selector_candidate_metrics.csv"
    )
    selection = pd.read_csv(OUTPUT_DIR / "selector_selection_rate.csv")
    importance = pd.read_csv(
        OUTPUT_DIR / "feature_importance_by_objective_fold.csv"
    )
    display(metrics)
    display(candidate_metrics)
    display(
        selection.groupby("candidate_id", as_index=False)["selected_rows"]
        .sum()
        .sort_values("selected_rows", ascending=False)
    )
    top_importance = (
        importance.groupby("feature", as_index=False)["gain_importance"]
        .mean()
        .sort_values("gain_importance", ascending=False)
        .head(30)
    )
    display(top_importance)
    if len(top_importance):
        axis = top_importance.sort_values("gain_importance").plot.barh(
            x="feature",
            y="gain_importance",
            figsize=(10, 10),
            legend=False,
            title="exp414 pred_abs_error mean gain importance",
        )
        axis.set_xlabel("mean gain importance across outer folds")
        plt.tight_layout()
        plt.savefig(
            OUTPUT_DIR / "feature_importance_pred_abs_error_top30.png",
            dpi=140,
        )
        plt.show()

    print(f"Elapsed seconds: {time.perf_counter() - started:.3f}")
    print("Generated files")
    for generated in sorted(OUTPUT_DIR.rglob("*")):
        if generated.is_file():
            print(generated.relative_to(OUTPUT_DIR), generated.stat().st_size)
