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
# # exp493 Student-t exact-HMM full replacement on exp264 — train
#
# exp264の固定12候補ID・順序・legal domainを変えず、Gaussian `exact_hmm`
# semantic slotをSHA固定済みexp374 Student-t exact HMMへ全面置換する。
# `exact_hmm`依存の2 pairと固定3-wayを再計算し、残り8候補はbitwise parityを
# 必須とする。exp264 corrected Stage Aの88列とcompact 74列は再選択せず、
# 同じ名前・順序でstrict nested dual selectorだけを再学習する。
#
# 実行量は1 variant / 2 objectives / outer 5 x inner 4 = 40 CPU boosters。
# saved exp264 controlは再学習しない。downstream TVT、inference、submissionは
# 実装・実行しない。

# %% [markdown]
# ## Contents
#
# 1. Imports and immutable experiment boundary
# 2. Notebook-safe runtime and path helpers
# 3. Authorization, candidate, and compute contract
# 4. Frozen inputs and SHA checks
# 5. Fixed12 replacement cache and Stage A schema rebuild
# 6. Stage C strict-nested dual selector
# 7. Scientific readout against saved exp264
# 8. Feature importance and generated artifacts
# 9. Reproducibility summary and fixed stop

# %% [markdown]
# ## 1. Imports and immutable experiment boundary

# %%
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
from IPython.display import display

from src.candidate_selector_pipeline import (
    audit_raw_context_availability,
    candidate_ids,
    read_yaml,
    resolve_exp263_cache_root,
    run_stage_c,
    sha256_file,
    write_json,
)
from src.exact_hmm_full_replacement import (
    CHANGED_CANDIDATES,
    Exp374Fixed12ReplacementCache,
    UNCHANGED_CANDIDATES,
    build_fixed12_replacement_readout,
    load_student_t_replacement_predictions,
    replacement_cost_contract,
    resolve_file_by_sha,
    resolve_parent_score_file,
    run_fixed12_stage_a_rebuild,
    stage_c_runtime_config,
    validate_fixed12_replacement_contract,
    write_fixed12_input_contract,
)

EXPERIMENT_NAME = "exp493_student_t_exact_hmm_full_replacement_on_exp264"
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")


def in_notebook_runtime() -> bool:
    try:
        return get_ipython() is not None  # type: ignore[name-defined]
    except NameError:
        return False


EXECUTE_NOTEBOOK = (
    os.environ.get("EXP493_IMPORT_ONLY", "0") != "1"
    and in_notebook_runtime()
)

# %% [markdown]
# ## 2. Notebook-safe runtime and path helpers

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
    candidates = [experiment_dir() / filename, Path.cwd() / filename]
    for path in candidates:
        if path.exists():
            return path
    matches = sorted(Path.cwd().rglob(filename))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(
        f"{filename} did not resolve uniquely: {matches}"
    )


def runtime_output_dir() -> Path:
    if in_notebook_runtime() and KAGGLE_WORKING_ROOT.exists():
        path = KAGGLE_WORKING_ROOT / "artifacts"
    else:
        path = experiment_dir() / "artifacts"
    path.mkdir(parents=True, exist_ok=True)
    return path


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


CONFIG = read_yaml(find_support_file("config.yaml"))
CONTRACT = read_yaml(find_support_file("candidate_contract.yaml"))
OUTPUT_DIR = runtime_output_dir()

# %% [markdown]
# ## 3. Authorization, candidate, and compute contract
#
# 実装承認とKaggle実行承認を分離する。canonical採用、package、runと
# 固定scopeがすべて承認済みの場合だけ40 booster学習を開始する。

# %%
contract_evidence = validate_fixed12_replacement_contract(CONTRACT)
cost_contract = replacement_cost_contract(CONFIG)
expected_approval_scope = (
    "fixed12_student_t_replacement_stage_a_stage_c_"
    "1_variant_2_objectives_outer5_inner4_40_cpu_boosters_"
    "no_control_retraining"
)
expected_v3_rerun_approval_scope = (
    "fixed12_student_t_v3_postprocess_fix_additional40_"
    "cumulative80_cpu_boosters_no_control_retraining"
)
assert CONFIG["experiment"]["route"] == "ensemble"
assert CONFIG["authorization"]["implementation_approved"] is True
assert CONFIG["authorization"]["canonical_notebook_adoption_approved"] is True
assert CONFIG["authorization"]["kaggle_package_approved"] is True
assert CONFIG["authorization"]["kaggle_run_approved"] is True
assert candidate_ids(CONTRACT) == contract_evidence["candidate_order"]
assert tuple(contract_evidence["changed_candidates"]) == CHANGED_CANDIDATES
assert tuple(contract_evidence["unchanged_candidates"]) == UNCHANGED_CANDIDATES
assert (
    contract_evidence["replacement_value_source"]
    == "exp374_student_t_df4_exact_hmm"
)
assert CONFIG["replacement"]["replacement_source_id"] == "student_t_exact_hmm"
display(
    {
        "experiment": EXPERIMENT_NAME,
        "route": CONFIG["experiment"]["route"],
        "status": CONFIG["experiment"]["status"],
        "candidate_order": contract_evidence["candidate_order"],
        "changed_candidates": contract_evidence["changed_candidates"],
        "unchanged_candidates": contract_evidence["unchanged_candidates"],
        "execution": cost_contract,
        "implementation_approved": True,
        "canonical_notebook_adopted": True,
        "kaggle_run_approved": CONFIG["execution"]["run_approved"],
        "kaggle_v3_rerun_approved": CONFIG["authorization"][
            "kaggle_v3_rerun_approved"
        ],
        "v3_additional_cpu_selector_boosters": CONFIG["execution"][
            "v3_rerun_additional_cpu_selector_boosters"
        ],
        "cumulative_cpu_selector_boosters": CONFIG["execution"][
            "v3_rerun_cumulative_cpu_selector_boosters"
        ],
        "downstream_tvt": False,
        "inference": False,
        "submission": False,
    }
)

if EXECUTE_NOTEBOOK:
    if not bool(CONFIG["authorization"]["kaggle_run_approved"]):
        raise RuntimeError(
            "exp493 Kaggle train is not authorized. Confirm 1 variant / "
            "2 objectives / outer5 x inner4 / 40 CPU boosters / saved control "
            "retraining 0, then record separate package/run approval."
        )
    if not bool(CONFIG["execution"]["run_approved"]):
        raise RuntimeError("exp493 execution.run_approved is false")
    if CONFIG["execution"].get("approved_scope") != expected_approval_scope:
        raise RuntimeError("exp493 approved scope differs from the frozen cost")
    if not bool(CONFIG["authorization"]["kaggle_v3_rerun_approved"]):
        raise RuntimeError("exp493 version 3 rerun is not authorized")
    if (
        CONFIG["execution"].get("v3_rerun_approved_scope")
        != expected_v3_rerun_approval_scope
    ):
        raise RuntimeError(
            "exp493 version 3 approved scope differs from the frozen "
            "additional/cumulative cost"
        )

# %% [markdown]
# ## 4. Frozen inputs and SHA checks
#
# exp374はallowlist 6列だけを読み、gzip decompressed SHAとpost-read logical SHAを
# 検証する。truth、error、gate、Gaussian control、source foldは読まない。
# parent scoreとhidden-like assignmentはselector予測後の比較専用である。

# %%
if EXECUTE_NOTEBOOK:
    started = time.perf_counter()
    roots = search_roots()
    raw_train_dir = find_raw_split("train")
    raw_test_dir = find_raw_split("test")

    parent_spec = CONFIG["data"]["parent_exp264"]
    parent_config_path = resolve_file_by_sha(
        parent_spec["config_patterns"],
        roots,
        expected_file_sha256=parent_spec["config_sha256"],
        label="exp264 parent config",
    )
    parent_config = read_yaml(parent_config_path)
    cache_root = resolve_exp263_cache_root(parent_config, roots)

    parent_feature_schema_path = resolve_file_by_sha(
        parent_spec["feature_schema_patterns"],
        roots,
        expected_file_sha256=parent_spec[
            "stage_a_feature_schema_file_sha256"
        ],
        label="exp264 corrected Stage A feature schema",
    )
    parent_feature_catalog_path = resolve_file_by_sha(
        parent_spec["feature_catalog_patterns"],
        roots,
        expected_file_sha256=parent_spec[
            "stage_a_feature_catalog_file_sha256"
        ],
        label="exp264 corrected Stage A feature catalog",
    )
    parent_compact_schema_path = resolve_file_by_sha(
        parent_spec["compact_schema_patterns"],
        roots,
        expected_file_sha256=parent_spec[
            "stage_c_compact_schema_file_sha256"
        ],
        label="exp264 corrected Stage C compact schema",
    )
    parent_score_path = resolve_parent_score_file(
        parent_spec["stage_c_outer_valid_score_patterns"],
        roots,
        parent_spec["stage_c_outer_valid_score_file_sha256"],
    )

    replacement_spec = CONFIG["data"]["replacement_prediction"]
    replacement_path = resolve_file_by_sha(
        replacement_spec["patterns"],
        roots,
        expected_file_sha256=replacement_spec["expected_file_sha256"],
        expected_decompressed_sha256=replacement_spec[
            "expected_decompressed_sha256"
        ],
        label="saved exp374 Student-t exact-HMM predictions",
    )
    replacement_predictions, replacement_manifest = (
        load_student_t_replacement_predictions(
            replacement_path,
            expected_rows=int(CONFIG["validation"]["expected_rows"]),
            expected_wells=int(CONFIG["validation"]["expected_wells"]),
            expected_file_sha256=replacement_spec["expected_file_sha256"],
            expected_decompressed_sha256=replacement_spec[
                "expected_decompressed_sha256"
            ],
            expected_post_read_prediction_sha256=replacement_spec[
                "expected_post_read_prediction_sha256"
            ],
        )
    )

    hidden_spec = CONFIG["data"]["hidden_like_assignment"]
    hidden_like_path = resolve_file_by_sha(
        hidden_spec["patterns"],
        roots,
        expected_file_sha256=hidden_spec["expected_sha256"],
        label="exp115 hidden-like fold assignment",
    )
    input_paths = {
        "raw_train_dir": str(raw_train_dir),
        "raw_test_dir": str(raw_test_dir),
        "exp263_cache_root": str(cache_root),
        "parent_config": {
            "path": str(parent_config_path),
            "sha256": sha256_file(parent_config_path),
        },
        "parent_feature_schema": {
            "path": str(parent_feature_schema_path),
            "sha256": sha256_file(parent_feature_schema_path),
        },
        "parent_feature_catalog": {
            "path": str(parent_feature_catalog_path),
            "sha256": sha256_file(parent_feature_catalog_path),
        },
        "parent_compact_schema": {
            "path": str(parent_compact_schema_path),
            "sha256": sha256_file(parent_compact_schema_path),
        },
        "parent_score": {
            "path": str(parent_score_path),
            "sha256": sha256_file(parent_score_path),
        },
        "replacement": replacement_manifest,
        "hidden_like_assignment": {
            "path": str(hidden_like_path),
            "sha256": sha256_file(hidden_like_path),
        },
    }
    display(input_paths)

# %% [markdown]
# ## 5. Fixed12 replacement cache and Stage A schema rebuild
#
# Student-t予測をglobal `(well_id,row_idx)` joinしてからexp263 outer foldへ再分割する。
# `exact_hmm`、K16×HMM、LikPF×HMM、固定3-wayだけをfloat32式で再計算する。
# 8 unchanged candidateは値・availability完全parityを要求する。
# frozen 88/74 schemaに対して各fold probeを作り、列再選択は行わない。

# %%
if EXECUTE_NOTEBOOK:
    replacement_cache = Exp374Fixed12ReplacementCache(
        cache_root,
        CONTRACT,
        exp374_predictions=replacement_predictions,
        exp374_manifest=replacement_manifest,
    )

    def replacement_cache_factory(
        requested_root: Path,
        requested_contract: dict[str, Any],
    ) -> Exp374Fixed12ReplacementCache:
        if Path(requested_root) != Path(cache_root):
            raise ValueError("replacement cache root changed after input freeze")
        if candidate_ids(requested_contract) != candidate_ids(CONTRACT):
            raise ValueError(
                "replacement candidate order changed after input freeze"
            )
        return replacement_cache

    write_fixed12_input_contract(
        OUTPUT_DIR / "exp493_input_contract.json",
        config=CONFIG,
        contract=CONTRACT,
        replacement_manifest=replacement_manifest,
        parent_score_path=parent_score_path,
    )
    raw_availability = audit_raw_context_availability(
        raw_train_dir,
        raw_test_dir,
        parent_config["features"]["raw_context"][
            "horizontal_numeric_allowlist"
        ],
    )
    raw_availability.to_csv(
        OUTPUT_DIR / "raw_context_availability_audit.csv", index=False
    )
    stage_a = run_fixed12_stage_a_rebuild(
        config=CONFIG,
        parent_config=parent_config,
        contract=CONTRACT,
        cache=replacement_cache,
        raw_train_dir=raw_train_dir,
        parent_feature_schema_path=parent_feature_schema_path,
        parent_feature_catalog_path=parent_feature_catalog_path,
        parent_compact_schema_path=parent_compact_schema_path,
        output_dir=OUTPUT_DIR,
    )
    assert stage_a["feature_count"] == 88
    assert stage_a["compact_feature_count"] == 74
    assert stage_a["models_trained"] == 0
    assert stage_a["truth_rows_loaded_before_feature_freeze"] == 0
    display(stage_a)

# %% [markdown]
# ## 6. Stage C strict-nested dual selector
#
# outer-train compactはinner OOF、outer-valid compactは4 inner model ensembleで
# 作る。2 objectives × outer 5 × inner 4 = 40 CPU boostersだけを学習し、
# saved exp264 selector/controlは再学習しない。

# %%
if EXECUTE_NOTEBOOK:
    runtime_config = stage_c_runtime_config(CONFIG, parent_config)
    stage_c = run_stage_c(
        config=runtime_config,
        contract=CONTRACT,
        cache_root=cache_root,
        raw_train_dir=raw_train_dir,
        output_dir=OUTPUT_DIR,
        cache_factory=replacement_cache_factory,
        hard_readout_enabled=True,
    )
    overlay_manifest = replacement_cache.replacement_manifest(
        expected_rows=int(CONFIG["validation"]["expected_rows"])
    )
    technical_cfg = CONFIG["guards"]["technical"]
    technical_checks = {
        "candidate_order_fixed12": candidate_ids(CONTRACT)
        == contract_evidence["candidate_order"],
        "changed_candidates_exactly_four": overlay_manifest["checks"][
            "changed_candidates_exactly_four"
        ],
        "unchanged_candidate_value_parity": overlay_manifest["checks"][
            "unchanged_candidate_value_parity"
        ],
        "unchanged_candidate_availability_parity": overlay_manifest["checks"][
            "unchanged_candidate_availability_parity"
        ],
        "replacement_formula_parity": overlay_manifest["checks"][
            "replacement_formula_parity"
        ],
        "global_key_parity": overlay_manifest["checks"][
            "global_key_join_rows_match"
        ],
        "truth_late": overlay_manifest["checks"][
            "truth_or_error_columns_loaded_before_feature_freeze"
        ],
        "replacement_source_fold_not_used": overlay_manifest["checks"][
            "source_fold_not_used_as_model_feature"
        ],
        "feature_schema_file_parity": sha256_file(
            OUTPUT_DIR / "feature_schema.json"
        )
        == parent_spec["stage_a_feature_schema_file_sha256"],
        "compact_schema_file_parity": sha256_file(
            OUTPUT_DIR / "compact_meta_schema.json"
        )
        == parent_spec["stage_c_compact_schema_file_sha256"],
        "model_count": int(stage_c["model_count"])
        == int(technical_cfg["expected_models"]),
        "compact_partition_count": int(stage_c["compact_partition_count"])
        == int(technical_cfg["expected_compact_partitions"]),
        "compact_rows": int(stage_c["compact_rows"])
        == int(technical_cfg["expected_compact_rows"]),
        "outer_valid_score_long_rows": int(
            stage_c["outer_valid_score_long_rows"]
        )
        == int(technical_cfg["expected_outer_valid_score_long_rows"]),
        "selector_score_guard": bool(stage_c["score_guard"]["passed"]),
        "nested_leakage_audit": bool(stage_c["leakage_audit"]["passed"]),
    }
    if not all(technical_checks.values()):
        raise RuntimeError(
            f"exp493 technical checks failed: {technical_checks}"
        )
    display(
        {
            "stage_c": stage_c,
            "technical_checks": technical_checks,
            "overlay_manifest": overlay_manifest,
        }
    )

# %% [markdown]
# ## 7. Scientific readout against saved exp264
#
# selector予測を凍結した後だけsaved exp264 hard score、hidden-like assignment、
# truth由来actual errorを比較する。primary 11候補hard RMSEを科学gateに使い、
# Student-t依存4候補の利用率とfixed fallback 7候補はreport-onlyとする。

# %%
if EXECUTE_NOTEBOOK:
    gate = build_fixed12_replacement_readout(
        new_score_path=(
            OUTPUT_DIR / "nested_outer_valid_candidate_score.parquet"
        ),
        parent_score_path=parent_score_path,
        hidden_like_assignment_path=hidden_like_path,
        contract=CONTRACT,
        score_summary=stage_c,
        technical_checks=technical_checks,
        saved_control=CONFIG["validation"]["saved_control"],
        guard_config=CONFIG["guards"]["scientific"],
        output_dir=OUTPUT_DIR,
        artifact_prefix="exp493",
    )
    decision = (
        "PASS_FIXED12_STUDENT_T_REPLACEMENT_SELECTOR"
        if bool(gate["passed"])
        else "FAIL_CLOSE_FIXED12_STUDENT_T_REPLACEMENT_SELECTOR"
    )
    display({"decision": decision, "scientific_gate": gate})

# %% [markdown]
# ## 8. Feature importance and generated artifacts
#
# fold/objective別gainを平均し、dual selectorが置換依存featureをどの程度使ったかを
# 人間が追える形で保存する。importanceは事後診断でありfeature再選択には使わない。

# %%
if EXECUTE_NOTEBOOK:
    import matplotlib.pyplot as plt

    importance_path = (
        OUTPUT_DIR
        / "nested_feature_importance_by_objective_outer_inner.csv"
    )
    importance = pd.read_csv(importance_path)
    required_importance_columns = {
        "objective",
        "feature",
        "importance_type",
        "importance",
    }
    missing_importance_columns = (
        required_importance_columns - set(importance.columns)
    )
    if missing_importance_columns:
        raise RuntimeError(
            "nested feature importance schema mismatch: "
            f"{sorted(missing_importance_columns)}"
        )
    gain_importance = importance.loc[
        importance["importance_type"].eq("gain")
    ].copy()
    if gain_importance.empty:
        raise RuntimeError("nested feature importance has no gain rows")
    importance_mean = (
        gain_importance.groupby(["objective", "feature"], as_index=False)[
            "importance"
        ]
        .mean()
        .rename(columns={"importance": "mean_gain_importance"})
        .sort_values(
            ["objective", "mean_gain_importance"],
            ascending=[True, False],
        )
    )
    importance_mean.to_csv(
        OUTPUT_DIR / "exp493_feature_importance_mean.csv", index=False
    )
    for objective, part in importance_mean.groupby("objective", sort=True):
        top = part.head(30).sort_values("mean_gain_importance")
        ax = top.plot.barh(
            x="feature",
            y="mean_gain_importance",
            figsize=(10, 11),
            legend=False,
            title=f"exp493 {objective} mean gain importance",
        )
        ax.set_xlabel("mean gain across outer/inner models")
        plt.tight_layout()
        plt.savefig(
            OUTPUT_DIR / f"exp493_{objective}_feature_importance_top30.png",
            dpi=140,
        )
        plt.show()
    display(importance_mean.groupby("objective", sort=True).head(30))
    print("Generated files")
    for generated in sorted(OUTPUT_DIR.rglob("*")):
        if generated.is_file():
            print(generated.relative_to(OUTPUT_DIR), generated.stat().st_size)

# %% [markdown]
# ## 9. Reproducibility summary and fixed stop
#
# 初回runはdeterministic anchorに昇格しない。PASS/FAILにかかわらず、
# downstream TVT、current-test inference、submissionはこの実験runで行わない。

# %%
if EXECUTE_NOTEBOOK:
    summary = {
        "status": "kaggle_cpu_stage_a_stage_c_completed",
        "decision": decision,
        "scientific_gate_passed": bool(gate["passed"]),
        "execution": cost_contract,
        "rows": int(CONFIG["validation"]["expected_rows"]),
        "wells": int(CONFIG["validation"]["expected_wells"]),
        "candidate_count": 12,
        "changed_candidate_count": 4,
        "unchanged_candidate_count": 8,
        "selector_feature_count": 88,
        "compact_feature_count": 74,
        "stage_a": stage_a,
        "stage_c": stage_c,
        "technical_checks": technical_checks,
        "scientific_gate": gate,
        "replacement_predictions": replacement_manifest,
        "replacement_overlay": overlay_manifest,
        "parent_exp264_score_sha256": sha256_file(parent_score_path),
        "elapsed_seconds": time.perf_counter() - started,
        "parent_control_retraining": False,
        "downstream_tvt_training": False,
        "inference": False,
        "submission": False,
        "deterministic_anchor": False,
    }
    write_json(OUTPUT_DIR / "exp493_summary.json", summary)
    reproducibility_path = OUTPUT_DIR / "reproducibility_manifest.json"
    reproducibility = json.loads(reproducibility_path.read_text())
    reproducibility.update(
        {
            "status": summary["status"],
            "decision": decision,
            "replacement_predictions": replacement_manifest,
            "replacement_overlay": overlay_manifest,
            "parent_exp264_score_sha256": summary[
                "parent_exp264_score_sha256"
            ],
            "scientific_gate": gate,
            "nested_selector_model_manifest_sha256": stage_c[
                "nested_selector_model_manifest_sha256"
            ],
            "nested_outer_valid_candidate_score_sha256": stage_c[
                "nested_outer_valid_candidate_score_sha256"
            ],
            "exp493_summary_sha256": sha256_file(
                OUTPUT_DIR / "exp493_summary.json"
            ),
        }
    )
    write_json(reproducibility_path, reproducibility)
    if KAGGLE_WORKING_ROOT.exists():
        write_json(KAGGLE_WORKING_ROOT / "metrics.json", summary)
    print("FINAL_SUMMARY", json.dumps(summary, sort_keys=True))
    print("Parent/control retrained:", False)
    print("Downstream TVT trained:", False)
    print("Inference executed:", False)
    print("Submission generated or submitted:", False)
