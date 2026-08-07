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
# # exp407 fold-safe inverse-RMSE weighted dual selector on exp264 — train
#
# corrected exp264 Stage B v5の12候補、88列raw-test-safe schema、outer 5 folds、
# deterministic sampling、2 objectives、LightGBM設定を固定し、各outer modelの正確な
# fit partitionだけで求めた候補別inverse-RMSE task weightをtraining rowsへ適用する。
# validation、early stopping、OOF metric、親比較gateはすべてunweightedのままとする。
#
# このNotebookは承認済みStage Bの正規Notebook採用元であり、
# `execution.run_approved=false`の間はfail closedする。
# Stage C、Stage D、inference、submissionは実装・実行しない。

# %% [markdown]
# ## Contents
#
# 1. Imports and immutable experiment boundary
# 2. Runtime, path, and serialization helpers
# 3. Candidate, compute, leakage, and weight contract
# 4. Pinned input resolution and SHA checks
# 5. Stage A raw-test-safe feature freeze
# 6. Fold-safe inverse-RMSE weighted Stage B
# 7. Parent v5 comparison and all-AND gate
# 8. Metrics, diagnostics, feature importance, and generated artifacts

# %% [markdown]
# ## 1. Imports and immutable experiment boundary

# %%
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pandas as pd

from src.candidate_selector_pipeline import (
    audit_raw_context_availability,
    read_yaml,
    resolve_exp263_cache_root,
    run_stage_a,
    run_stage_b,
    sha256_file,
)
from src.exp407_inverse_rmse_selector import (
    EXPERIMENT_NAME,
    evaluate_exp407_stage_b,
    resolve_pinned_input,
    validate_exp407_static_contract,
)

KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")
APPROVED_STAGE_B_SCOPE = (
    "inverse_rmse_weighted_stage_b_1_variant_2_objectives_"
    "5_outer_10_cpu_boosters_no_control_retraining"
)


def in_notebook_runtime() -> bool:
    try:
        return get_ipython() is not None  # type: ignore[name-defined]
    except NameError:
        return False


EXECUTE_NOTEBOOK = (
    os.environ.get("EXP407_IMPORT_ONLY", "0") != "1" and in_notebook_runtime()
)
if EXECUTE_NOTEBOOK:
    import matplotlib.pyplot as plt
    from IPython.display import display

# %% [markdown]
# ## 2. Runtime, path, and serialization helpers
#
# Notebook-safeなcwd探索を使い、bootstrap後の`/kaggle/working`、repo root、
# 実験ディレクトリの順でsupport fileを解決する。

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
    direct = [
        Path.cwd() / filename,
        experiment_dir() / filename,
        KAGGLE_WORKING_ROOT / filename,
    ]
    for path in direct:
        if path.is_file():
            return path
    experiment_matches = sorted(
        path
        for path in Path.cwd().rglob(filename)
        if EXPERIMENT_NAME in path.parts
    )
    if len(experiment_matches) == 1:
        return experiment_matches[0]
    raise FileNotFoundError(
        f"{filename} did not resolve uniquely for {EXPERIMENT_NAME}: "
        f"{experiment_matches}"
    )


def runtime_output_dir() -> Path:
    if EXECUTE_NOTEBOOK and KAGGLE_WORKING_ROOT.exists():
        return KAGGLE_WORKING_ROOT / "artifacts"
    return experiment_dir() / "artifacts"


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
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"raw {split} directory did not resolve: {matches}")


def search_roots() -> list[Path]:
    return [
        Path.cwd(),
        project_root(),
        KAGGLE_INPUT_ROOT,
        Path("/tmp/kaggle-output"),
    ]


CONFIG_PATH = find_support_file("config.yaml")
CONTRACT_PATH = find_support_file("candidate_contract.yaml")
CONFIG = read_yaml(CONFIG_PATH)
CONTRACT = read_yaml(CONTRACT_PATH)
OUTPUT_DIR = runtime_output_dir()
if sha256_file(CONTRACT_PATH) != str(CONFIG["data"]["candidate_contract_sha256"]):
    raise ValueError("exp407 candidate contract SHA differs from the frozen parent copy")

# %% [markdown]
# ## 3. Candidate, compute, leakage, and weight contract
#
# 実行量は1 variant × 2 objectives × 5 outer folds = 10 CPU boosters。
# 保存済みparent v5をcontrolとして読み、control再学習、PF/HMM/Beam再生成、GPU学習は0。
# weight関数にはfit labelsだけを渡し、outer-valid truth、global OOF truth、
# hidden-like roleはweight生成後の評価にだけ使う。

# %%
static_contract = validate_exp407_static_contract(CONFIG, CONTRACT)
expected_candidate_order = [str(item) for item in CONFIG["candidate_bank"]["order"]]
assert static_contract["candidate_order"] == expected_candidate_order
assert static_contract["candidate_count"] == 12
assert static_contract["legal_domain_counts"] == [11, 7]
assert static_contract["feature_count"] == 88
assert static_contract["cost"] == {
    "active_variants": 1,
    "objectives": 2,
    "outer_folds": 5,
    "planned_cpu_boosters": 10,
    "parent_control_retraining": False,
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
            "candidate_order": expected_candidate_order,
            "feature_schema_sha256": CONFIG["features"][
                "feature_schema_logical_sha256"
            ],
            "cost": static_contract["cost"],
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
# ## 4. Pinned input resolution and SHA checks
#
# Stage A/Bの入力はexp263 Stage 0 cacheとexp251 v4 schema。科学比較では保存済み
# corrected exp264 Stage B v5のmetrics / distance bucket / by-wellだけを読み、
# hidden-like assignmentはfit完了後のsubgroup readoutにだけ使う。

# %%
if EXECUTE_NOTEBOOK:
    if not bool(CONFIG["execution"]["run_approved"]):
        raise RuntimeError(
            "exp407 Kaggle Stage B execution is not approved. Keep "
            "execution.run_approved=false until the user separately approves "
            "1 variant / 2 objectives / 5 outer folds / 10 CPU boosters / "
            "parent-control retraining 0."
        )
    if CONFIG["execution"]["approval_scope"] != APPROVED_STAGE_B_SCOPE:
        raise RuntimeError("exp407 Stage B approval scope does not match the fixed cost")

    started = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    roots = search_roots()
    raw_train_dir = find_raw_split("train")
    raw_test_dir = find_raw_split("test")
    cache_root = resolve_exp263_cache_root(CONFIG, roots)
    parent_schema_path = resolve_pinned_input(
        CONFIG["data"]["exp251_selected_feature_schema_patterns"],
        roots,
        expected_sha256=CONFIG["data"]["exp251_selected_feature_schema_sha256"],
        label="exp251 v4 selected feature schema",
    )
    parent_cfg = CONFIG["data"]["parent_stage_b_v5"]
    parent_metrics_path = resolve_pinned_input(
        parent_cfg["selector_metrics_patterns"],
        roots,
        expected_sha256=parent_cfg["selector_metrics_sha256"],
        label="corrected exp264 Stage B v5 selector metrics",
    )
    parent_bucket_path = resolve_pinned_input(
        parent_cfg["distance_bucket_metrics_patterns"],
        roots,
        expected_sha256=parent_cfg["distance_bucket_metrics_sha256"],
        label="corrected exp264 Stage B v5 distance metrics",
    )
    parent_by_well_path = resolve_pinned_input(
        parent_cfg["by_well_metrics_patterns"],
        roots,
        expected_sha256=parent_cfg["by_well_metrics_sha256"],
        label="corrected exp264 Stage B v5 by-well metrics",
    )
    hidden_cfg = CONFIG["data"]["hidden_like_assignment"]
    hidden_like_assignment_path = resolve_pinned_input(
        hidden_cfg["patterns"],
        roots,
        expected_sha256=hidden_cfg["sha256"],
        label="exp115 hidden-like assignment",
    )
    input_contract = {
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
        "parent_selector_metrics": {
            "path": str(parent_metrics_path),
            "sha256": sha256_file(parent_metrics_path),
        },
        "parent_distance_metrics": {
            "path": str(parent_bucket_path),
            "sha256": sha256_file(parent_bucket_path),
        },
        "parent_by_well": {
            "path": str(parent_by_well_path),
            "sha256": sha256_file(parent_by_well_path),
        },
        "hidden_like_assignment": {
            "path": str(hidden_like_assignment_path),
            "sha256": sha256_file(hidden_like_assignment_path),
        },
    }
    print(json.dumps(input_contract, indent=2))

# %% [markdown]
# ## 5. Stage A raw-test-safe feature freeze
#
# 親と同じraw `MD/X/Y/Z/GR`、shape windows、candidate/family/formula one-hotから
# feature auditを再構成する。selected 88列とlogical SHAがcorrected exp264 v4から
# 1列でも変われば、weight計算やfitへ進まない。

# %%
if EXECUTE_NOTEBOOK:
    availability = audit_raw_context_availability(
        raw_train_dir,
        raw_test_dir,
        CONFIG["features"]["raw_context"]["horizontal_numeric_allowlist"],
    )
    availability.to_csv(
        OUTPUT_DIR / "raw_context_availability_audit.csv",
        index=False,
    )
    display(availability)
    if not bool(availability["availability_pass"].all()):
        raise RuntimeError("raw train/test context availability audit failed")

    stage_a = run_stage_a(
        config=CONFIG,
        contract=CONTRACT,
        cache_root=cache_root,
        raw_train_dir=raw_train_dir,
        output_dir=OUTPUT_DIR,
        parent_schema_path=parent_schema_path,
    )
    if int(stage_a["selected_feature_count"]) != 88:
        raise RuntimeError("exp407 Stage A did not reproduce 88 selected features")
    if stage_a["feature_schema_sha256"] != CONFIG["features"][
        "feature_schema_logical_sha256"
    ]:
        raise RuntimeError("exp407 Stage A feature schema differs from corrected exp264")
    display(stage_a)
    feature_catalog = pd.read_csv(OUTPUT_DIR / "feature_catalog.csv")
    display(
        feature_catalog.groupby(["group", "selected"], dropna=False)
        .size()
        .rename("features")
    )

# %% [markdown]
# ## 6. Fold-safe inverse-RMSE weighted Stage B
#
# shared Stage B pipelineが親と同じfit row IDsを作った後、candidate-long fit labelsだけを
# weight helperへ渡す。候補別RMSE、inverse、pre-clip normalize、clip、final weight、
# row/sample/content SHA、truth-read ledgerをfold別保存する。同じsample weightを
# `pred_abs_error` / `p_within10`のtrainingへ渡し、eval_setにはweightを渡さない。

# %%
if EXECUTE_NOTEBOOK:
    stage_b = run_stage_b(
        config=CONFIG,
        contract=CONTRACT,
        cache_root=cache_root,
        raw_train_dir=raw_train_dir,
        output_dir=OUTPUT_DIR,
        candidate_task_weight_config=CONFIG["candidate_task_weight"],
    )
    if int(stage_b["model_count"]) != 10:
        raise RuntimeError("exp407 Stage B did not produce exactly 10 selector models")
    if int(stage_b["candidate_score_oof_rows"]) != int(
        CONFIG["data"]["expected_candidate_long_oof_rows"]
    ):
        raise RuntimeError("exp407 candidate-score OOF row count changed")
    weight_manifest = stage_b["candidate_task_weight"]
    if not bool(weight_manifest["all_checks_passed"]):
        raise RuntimeError("exp407 candidate task weight audit failed")
    display(stage_b)
    display(pd.read_csv(OUTPUT_DIR / "candidate_task_weight_by_fold.csv"))
    display(pd.read_csv(OUTPUT_DIR / "candidate_task_weight_truth_read_ledger.csv"))

# %% [markdown]
# ## 7. Parent v5 comparison and all-AND gate
#
# 保存済みparent v5を唯一のcontrolとする。pooled/fold score、hard top1、near、
# 1000+、hidden-like 2面、worst-wellをnew-minus-parentで評価する。
# Technicalまたはscientificの1項目でもFAILならStage Cへ自動昇格しない。

# %%
if EXECUTE_NOTEBOOK:
    gate = evaluate_exp407_stage_b(
        config=CONFIG,
        contract=CONTRACT,
        output_dir=OUTPUT_DIR,
        parent_metrics_path=parent_metrics_path,
        parent_bucket_path=parent_bucket_path,
        parent_by_well_path=parent_by_well_path,
        hidden_like_assignment_path=hidden_like_assignment_path,
        source_config_path=CONFIG_PATH,
        source_candidate_contract_path=CONTRACT_PATH,
    )
    display(gate)
    display(pd.read_csv(OUTPUT_DIR / "exp407_parent_fold_comparison.csv"))
    display(pd.read_csv(OUTPUT_DIR / "exp407_parent_bucket_comparison.csv"))
    display(pd.read_csv(OUTPUT_DIR / "exp407_parent_hidden_like_comparison.csv"))
    print("Decision:", gate["decision"])

# %% [markdown]
# ## 8. Metrics, diagnostics, feature importance, and generated artifacts

# %%
if EXECUTE_NOTEBOOK:
    selector_metrics = pd.read_csv(OUTPUT_DIR / "selector_metrics.csv")
    candidate_metrics = pd.read_csv(OUTPUT_DIR / "selector_candidate_metrics.csv")
    selection = pd.read_csv(OUTPUT_DIR / "selector_selection_rate.csv")
    calibration = pd.read_csv(OUTPUT_DIR / "selector_calibration.csv")
    importance = pd.read_csv(
        OUTPUT_DIR / "feature_importance_by_objective_fold.csv"
    )
    display(selector_metrics)
    display(candidate_metrics)
    display(
        selection.groupby(["objective", "candidate_id"], as_index=False)[
            "selected_rows"
        ]
        .sum()
        .sort_values(["objective", "selected_rows"], ascending=[True, False])
    )
    display(calibration.head(100))

    importance_mean = (
        importance.groupby(["objective", "feature"], as_index=False)[
            "gain_importance"
        ]
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
                title=f"exp407 {objective} mean gain importance",
            )
            ax.set_xlabel("mean gain importance across outer folds")
            plt.tight_layout()
            plt.savefig(
                OUTPUT_DIR / f"feature_importance_{objective}_top30.png",
                dpi=140,
            )
            plt.show()

    print(f"Elapsed seconds: {time.perf_counter() - started:.3f}")
    print("Generated files")
    for generated in sorted(OUTPUT_DIR.rglob("*")):
        if generated.is_file():
            print(generated.relative_to(OUTPUT_DIR), generated.stat().st_size)
