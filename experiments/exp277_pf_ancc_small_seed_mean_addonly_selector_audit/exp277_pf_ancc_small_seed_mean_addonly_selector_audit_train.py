# %% [markdown]
# # exp277 PF ANCC small-seed mean add-only selector audit — train
#
# exp271 version 2の固定mean4/mean8 pathでexp263 core-12内のpf_anccを差し替え、
# exp264修正版のraw-test-only outer5-inner4 dual selector compactを再構築する。downstreamでは
# clean 273列へcompact 74列だけをadd-onlyし、保存済み修正版Stage D matched-control OOFと比較する。
# PF再生成、control再学習、hard top1、candidate平均、inference、submissionは行わない。

# %% [markdown]
# ## Contents
# 1. Imports and runtime helpers
# 2. Stage, variant, and compute contract
# 3. Input and SHA contracts
# 4. PF candidate augmentation contract
# 5. Nested selector or downstream execution
# 6. Metrics, guards, and feature importance
# 7. Generated artifacts and reproducibility evidence

# %% [markdown]
# ## 1. Imports and runtime helpers

# %%
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import yaml
from IPython.display import display

from src.candidate_selector_pipeline import (
    read_yaml,
    resolve_existing_path,
    resolve_exp263_cache_root,
    sha256_file,
    verify_exp263_root,
)
from src.pf_ancc_selector_audit import (
    VARIANTS,
    aggregate_downstream_variants,
    build_variant_contract,
    resolve_complete_nested_root,
    resolve_downstream_root,
    resolve_pf_candidate_source,
    run_downstream_variant,
    run_nested_selector_variant,
)

EXPERIMENT_NAME = "exp277_pf_ancc_small_seed_mean_addonly_selector_audit"
PACKAGE_DIR = Path.cwd()
if not (PACKAGE_DIR / "config.yaml").exists():
    PACKAGE_DIR = Path("experiments") / EXPERIMENT_NAME
CONFIG = yaml.safe_load((PACKAGE_DIR / "config.yaml").read_text())
OUTPUT_DIR = (
    Path("/kaggle/working/artifacts")
    if Path("/kaggle/working").exists()
    else PACKAGE_DIR / "artifacts"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SEARCH_ROOTS = [Path("/kaggle/input"), Path("/tmp"), Path.cwd()]


def find_raw_train_dir() -> Path:
    direct = Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction/train")
    if direct.is_dir():
        return direct
    for sample in sorted(Path("/kaggle/input").rglob("sample_submission.csv")):
        candidate = sample.parent / "train"
        if candidate.is_dir():
            return candidate
    local = Path("data/raw/train")
    if local.is_dir():
        return local
    raise FileNotFoundError("competition train directory was not found")


def find_raw_test_dir() -> Path:
    direct = Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction/test")
    if direct.is_dir():
        return direct
    for sample in sorted(Path("/kaggle/input").rglob("sample_submission.csv")):
        candidate = sample.parent / "test"
        if candidate.is_dir():
            return candidate
    local = Path("data/raw/test")
    if local.is_dir():
        return local
    raise FileNotFoundError("competition test directory was not found")


def require_kaggle_or_explicit_local() -> None:
    if Path("/kaggle/working").exists() or os.environ.get("EXPERIMENT_ALLOW_LOCAL") == "1":
        return
    raise RuntimeError("Kaggle Notebook is authoritative; local full execution is disabled")


def resolve_support_file(filename: str) -> Path:
    candidates = [PACKAGE_DIR / filename, Path.cwd() / filename]
    candidates.extend(Path("/kaggle/working").rglob(filename))
    match = next((path for path in candidates if path.exists()), None)
    if match is None:
        raise FileNotFoundError(filename)
    return match


# %% [markdown]
# ## 2. Stage, variant, and compute contract
#
# nested stageは1 variant × 2 objectives × 5 outer × 4 inner = 40 CPU boosters。
# downstream stageは1 variant × 3 configs × 5 outer = 15 GPU boostersで、control再学習は0。
# aggregateは保存済み3 OOFを読む0-booster stage。`run_approved=false`では学習stageを停止する。

# %%
STAGE = str(CONFIG["execution"]["stage"])
ACTIVE_VARIANT = str(CONFIG["execution"]["active_variant"])
RUN_APPROVED = bool(CONFIG["execution"]["run_approved"])
if STAGE not in set(CONFIG["execution"]["allowed_stages"]):
    raise ValueError(f"unknown execution stage: {STAGE}")
if ACTIVE_VARIANT not in VARIANTS:
    raise ValueError(f"unknown active variant: {ACTIVE_VARIANT}")
if STAGE.startswith("nested_selector_") or STAGE.startswith("downstream_"):
    expected_variant = STAGE.removeprefix("nested_selector_").removeprefix("downstream_")
    if expected_variant != ACTIVE_VARIANT:
        raise ValueError(f"stage/active_variant mismatch: {STAGE} / {ACTIVE_VARIANT}")
    if not RUN_APPROVED:
        raise RuntimeError(f"{STAGE} requires execution.run_approved=true after cost approval")

if STAGE.startswith("nested_selector_"):
    COST_CONTRACT = {
        "stage": STAGE,
        "active_variants": 1,
        "objectives": 2,
        "outer_folds": 5,
        "inner_folds": 4,
        "total_boosters": 40,
        "device": "cpu",
        "control_retraining": False,
    }
elif STAGE.startswith("downstream_"):
    COST_CONTRACT = {
        "stage": STAGE,
        "active_variants": 1,
        "configs": 3,
        "outer_folds": 5,
        "total_boosters": 15,
        "device": "gpu",
        "control_retraining": False,
    }
else:
    COST_CONTRACT = {
        "stage": STAGE,
        "active_variants": 0,
        "total_boosters": 0,
        "device": "cpu",
        "control_retraining": False,
    }
display(COST_CONTRACT)
assert CONFIG["experiment"]["route"] == "ensemble"
assert CONFIG["execution"]["parent_control_retraining"] is False
assert CONFIG["inference"]["enabled"] is False
assert CONFIG["inference"]["submit_to_kaggle"] is False

# %% [markdown]
# ## 3. Input and SHA contracts
#
# nested stageはexp263 60 partition、exp271 candidate gzip、exp251 schemaを読み、actual train/test
# headerでMD/X/Y/Z/GRのavailabilityを照合する。downstreamはvariant一致済みnested 25 partitions、
# 修正版exp264 fixed-control OOF、clean 273 allowlist、exp218 source/config、hidden-like assignmentを読む。

# %%
FULL_CONTRACT = read_yaml(resolve_support_file(str(CONFIG["data"]["candidate_contract"])))
VARIANT_CONTRACT = build_variant_contract(FULL_CONTRACT, ACTIVE_VARIANT)
display(
    {
        "experiment": EXPERIMENT_NAME,
        "stage": STAGE,
        "variant": ACTIVE_VARIANT,
        "route": CONFIG["experiment"]["route"],
        "candidate_order": [item["id"] for item in VARIANT_CONTRACT["score_candidates"]],
        "disagreement_enabled": VARIANT_CONTRACT["disagreement_enabled"],
    }
)
for rule in CONFIG["validation"]["leakage_policy"]:
    print("-", rule)

RAW_TRAIN_DIR: Path | None = None
RAW_TEST_DIR: Path | None = None
EXP263_ROOT: Path | None = None
PF_CANDIDATE_PATH: Path | None = None
PARENT_SCHEMA_PATH: Path | None = None
NESTED_ROOT: Path | None = None
FIXED_CONTROL_PATH: Path | None = None
EXP218_SOURCE_PATH: Path | None = None
EXP218_CONFIG_PATH: Path | None = None
BASE_FEATURE_ALLOWLIST_PATH: Path | None = None
HIDDEN_ASSIGNMENT_PATH: Path | None = None
if STAGE != "design_only":
    require_kaggle_or_explicit_local()
if STAGE.startswith("nested_selector_"):
    RAW_TRAIN_DIR = find_raw_train_dir()
    RAW_TEST_DIR = find_raw_test_dir()
    EXP263_ROOT = resolve_exp263_cache_root(CONFIG, SEARCH_ROOTS)
    PF_CANDIDATE_PATH = resolve_pf_candidate_source(CONFIG, SEARCH_ROOTS)
    PARENT_SCHEMA_PATH = resolve_existing_path(
        [str(item) for item in CONFIG["data"]["exp251_selected_feature_schema_patterns"]],
        SEARCH_ROOTS,
    )
    display(
        {
            "exp263": verify_exp263_root(EXP263_ROOT, CONFIG),
            "exp271_candidate": str(PF_CANDIDATE_PATH),
            "exp271_candidate_raw_sha256": sha256_file(PF_CANDIDATE_PATH),
            "parent_schema": str(PARENT_SCHEMA_PATH),
            "raw_train_dir": str(RAW_TRAIN_DIR),
            "raw_test_dir": str(RAW_TEST_DIR),
            "raw_context_allowlist": CONFIG["features"]["raw_context"][
                "horizontal_numeric_allowlist"
            ],
        }
    )
elif STAGE.startswith("downstream_"):
    RAW_TRAIN_DIR = find_raw_train_dir()
    patterns = CONFIG["data"]["nested_variant_root_patterns"][ACTIVE_VARIANT]
    NESTED_ROOT = resolve_complete_nested_root(patterns, SEARCH_ROOTS, ACTIVE_VARIANT)
    FIXED_CONTROL_PATH = resolve_existing_path(
        [str(item) for item in CONFIG["data"]["exp264_fixed_control_oof_patterns"]],
        SEARCH_ROOTS,
    )
    EXP218_SOURCE_PATH = resolve_existing_path(
        [str(item) for item in CONFIG["data"]["exp218_source_patterns"]], SEARCH_ROOTS
    )
    EXP218_CONFIG_PATH = resolve_existing_path(
        [str(item) for item in CONFIG["data"]["exp218_config_patterns"]], SEARCH_ROOTS
    )
    BASE_FEATURE_ALLOWLIST_PATH = resolve_existing_path(
        [
            str(item)
            for item in CONFIG["data"]["exp218_clean_273_allowlist_patterns"]
        ],
        SEARCH_ROOTS,
    )
    HIDDEN_ASSIGNMENT_PATH = resolve_existing_path(
        [str(item) for item in CONFIG["data"]["hidden_like_assignment_patterns"]],
        SEARCH_ROOTS,
    )
    display(
        {
            "nested_root": str(NESTED_ROOT),
            "fixed_control_oof": str(FIXED_CONTROL_PATH),
            "fixed_control_sha256": sha256_file(FIXED_CONTROL_PATH),
            "exp218_source": str(EXP218_SOURCE_PATH),
            "exp218_config": str(EXP218_CONFIG_PATH),
            "base_feature_allowlist": str(BASE_FEATURE_ALLOWLIST_PATH),
            "base_feature_allowlist_sha256": sha256_file(BASE_FEATURE_ALLOWLIST_PATH),
            "hidden_like_assignment": str(HIDDEN_ASSIGNMENT_PATH),
        }
    )

# %% [markdown]
# ## 4. PF candidate augmentation contract
#
# `mean4_only` / `mean8_only`は既存pf_ancc slotを1本のmean pathで置換し、12候補を維持する。
# `mean4_mean8_disagreement`だけがseed std4/8、particle std4/8、mean差signed/absoluteを
# candidate confidence blockへ追加し、13候補にする。候補・特徴量はvariant契約固定後に変更しない。

# %%
display(
    {
        "base_candidate_count": CONFIG["candidate_bank"]["base_candidate_count"],
        "replacement_target": CONFIG["candidate_bank"]["replacement_target"],
        "variant_contract": CONFIG["candidate_bank"]["variants"][ACTIVE_VARIANT],
        "score_candidate_count": len(VARIANT_CONTRACT["score_candidates"]),
        "hard_selection_enabled": CONFIG["candidate_bank"]["hard_selection_enabled"],
        "single_candidate_control": CONFIG["candidate_bank"]["single_candidate_control"],
    }
)

# %% [markdown]
# ## 5. Nested selector or downstream execution
#
# nested runはfeature auditと40-model leakage-free compact生成を同じvariant runで完結させる。
# downstream runはfixed controlを読むだけで、add-only 15 modelsだけを学習する。
# aggregateは3 downstream rootの内部manifest SHAを検証して比較表を作る。

# %%
SUMMARY: dict[str, Any] | None = None
if STAGE == "design_only":
    print("Design-only guard: no feature generation or model training was executed.")
elif STAGE.startswith("nested_selector_"):
    assert all(
        value is not None
        for value in [
            RAW_TRAIN_DIR,
            RAW_TEST_DIR,
            EXP263_ROOT,
            PF_CANDIDATE_PATH,
            PARENT_SCHEMA_PATH,
        ]
    )
    SUMMARY = run_nested_selector_variant(
        config=CONFIG,
        full_contract=FULL_CONTRACT,
        variant=ACTIVE_VARIANT,
        cache_root=EXP263_ROOT,
        pf_candidate_path=PF_CANDIDATE_PATH,
        raw_train_dir=RAW_TRAIN_DIR,
        raw_test_dir=RAW_TEST_DIR,
        output_dir=OUTPUT_DIR,
        parent_schema_path=PARENT_SCHEMA_PATH,
    )
elif STAGE.startswith("downstream_"):
    assert CONFIG["runtime"]["kaggle"]["enable_gpu"] is True
    assert all(
        value is not None
        for value in [
            RAW_TRAIN_DIR,
            NESTED_ROOT,
            FIXED_CONTROL_PATH,
            EXP218_SOURCE_PATH,
            EXP218_CONFIG_PATH,
            BASE_FEATURE_ALLOWLIST_PATH,
            HIDDEN_ASSIGNMENT_PATH,
        ]
    )
    SUMMARY = run_downstream_variant(
        config=CONFIG,
        full_contract=FULL_CONTRACT,
        variant=ACTIVE_VARIANT,
        nested_root=NESTED_ROOT,
        fixed_control_oof_path=FIXED_CONTROL_PATH,
        exp218_source_path=EXP218_SOURCE_PATH,
        exp218_config_path=EXP218_CONFIG_PATH,
        base_feature_allowlist_path=BASE_FEATURE_ALLOWLIST_PATH,
        hidden_like_assignment_path=HIDDEN_ASSIGNMENT_PATH,
        raw_train_dir=RAW_TRAIN_DIR,
        output_dir=OUTPUT_DIR,
    )
else:
    roots = {
        variant: resolve_downstream_root(
            CONFIG["data"]["downstream_variant_root_patterns"][variant],
            SEARCH_ROOTS,
            variant,
        )
        for variant in VARIANTS
    }
    SUMMARY = aggregate_downstream_variants(variant_roots=roots, output_dir=OUTPUT_DIR)
if SUMMARY is not None:
    display(SUMMARY)

# %% [markdown]
# ## 6. Metrics, guards, and feature importance
#
# nestedではobjective別mean gain、downstreamではPF/selector compact列のmean gainを表示する。
# 採用判定はoverall、3-of-5 folds、1000+、hidden-like 2面、worst-wellの全guardで行う。

# %%
if STAGE.startswith("nested_selector_") and (OUTPUT_DIR / "nested_selector_metrics.json").exists():
    nested_metrics = json.loads((OUTPUT_DIR / "nested_selector_metrics.json").read_text())
    display(nested_metrics)
    importance = pd.read_csv(
        OUTPUT_DIR / "nested_feature_importance_by_objective_outer_inner.csv"
    )
    mean_importance = (
        importance[importance["importance_type"].eq("gain")]
        .groupby(["objective", "feature"], as_index=False)["importance"]
        .mean()
        .sort_values(["objective", "importance"], ascending=[True, False])
    )
    for objective in ("pred_abs_error", "p_within10"):
        top = mean_importance[mean_importance["objective"].eq(objective)].head(30)
        display(top)
        ax = top.sort_values("importance").plot.barh(
            x="feature",
            y="importance",
            figsize=(10, 10),
            legend=False,
            title=f"exp277 {ACTIVE_VARIANT} {objective} mean gain",
        )
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"nested_{ACTIVE_VARIANT}_{objective}_top30.png", dpi=140)
        plt.show()
elif STAGE.startswith("downstream_") and (OUTPUT_DIR / "downstream_metrics.json").exists():
    downstream_metrics = json.loads((OUTPUT_DIR / "downstream_metrics.json").read_text())
    display(downstream_metrics)
    display(pd.read_csv(OUTPUT_DIR / "downstream_fold_metrics.csv"))
    display(pd.read_csv(OUTPUT_DIR / "downstream_bucket_metrics.csv"))
    display(pd.read_csv(OUTPUT_DIR / "downstream_hidden_like_metrics.csv"))
    importance = pd.read_csv(OUTPUT_DIR / "downstream_feature_importance.csv")
    compact = (
        importance[
            importance["importance_type"].eq("gain")
            & importance["feature"].str.startswith("selector__")
        ]
        .groupby("feature", as_index=False)["importance"]
        .mean()
        .sort_values("importance", ascending=False)
        .head(40)
    )
    display(compact)
    ax = compact.head(30).sort_values("importance").plot.barh(
        x="feature",
        y="importance",
        figsize=(10, 11),
        legend=False,
        title=f"exp277 {ACTIVE_VARIANT} compact mean gain",
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"downstream_{ACTIVE_VARIANT}_compact_top30.png", dpi=140)
    plt.show()
elif STAGE == "aggregate_compare" and (OUTPUT_DIR / "aggregate_variant_comparison.csv").exists():
    display(pd.read_csv(OUTPUT_DIR / "aggregate_variant_comparison.csv"))

# %% [markdown]
# ## 7. Generated artifacts and reproducibility evidence
#
# model manifest、OOF prediction、feature/compact schema、metricsのSHAを一覧化する。
# gzip入力はraw SHAに加えてdecompressed content SHAをnested summaryへ保存する。

# %%
generated = []
for path in sorted(OUTPUT_DIR.rglob("*")):
    if path.is_file():
        generated.append(
            {
                "path": str(path.relative_to(OUTPUT_DIR)),
                "bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
display(pd.DataFrame(generated))
print(
    json.dumps(
        {
            "status": "implemented_stage_completed" if SUMMARY is not None else "design_only",
            "stage": STAGE,
            "variant": ACTIVE_VARIANT,
            "cost_contract": COST_CONTRACT,
            "generated_files": len(generated),
        },
        sort_keys=True,
    )
)
