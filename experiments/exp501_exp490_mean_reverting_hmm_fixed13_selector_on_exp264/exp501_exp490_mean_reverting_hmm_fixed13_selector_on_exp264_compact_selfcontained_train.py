# %% [markdown]
# # exp501 exp490 mean-reverting HMM fixed13 selector on exp264 — train
#
# Add the saved exp490 geometry-centered mean-reverting exact-HMM prediction as
# exactly one new primary candidate to the corrected exp264 dual selector. The
# prediction and its two target-free state fields are SHA-frozen inputs. This
# notebook never reruns HMM/PF/Beam, the parent selector, downstream TVT,
# current-test generation, inference, or submission. Stage A refreezes the
# candidate-long schema, Stage C fits the unchanged outer-5 / inner-4 dual
# selector, and all truth-bearing scientific diagnostics run post-freeze.

# %% [markdown]
# ## Contents
# 1. Imports and immutable experiment boundary
# 2. Notebook-safe runtime and path helpers
# 3. Authorization, candidate, and compute contract
# 4. Frozen exp263 / exp490 / exp264 input checks
# 5. Fixed13 cache and Stage A feature freeze
# 6. Stage C strict-nested dual selector
# 7. Fixed13 versus fixed12 scientific and reranking readouts
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

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

from src.candidate_selector_pipeline import (
    audit_raw_context_availability,
    candidate_ids,
    compact_feature_names,
    read_yaml,
    resolve_existing_path,
    resolve_exp263_cache_root,
    run_stage_a,
    run_stage_c,
    sha256_file,
    write_json,
)
from src.exp490_fixed13_candidate_cache import (
    ADDED_CANDIDATE_ID,
    BASE_CANDIDATE_IDS,
    EXP490_NATIVE_FIELDS,
    Exp490Fixed13CandidateCache,
    build_fixed13_integration_readout,
    build_incumbent_reranking_diagnostic,
    build_postfreeze_addone_novelty_readout,
    load_exp490_target_free_inputs,
    pair_selector_scores,
    resolve_csv_by_payload_sha,
    resolve_file_by_sha,
    validate_fixed13_contract,
    write_exp501_input_contract,
)

EXPERIMENT_NAME = "exp501_exp490_mean_reverting_hmm_fixed13_selector_on_exp264"
OUTPUT_PREFIX = EXPERIMENT_NAME
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")


def in_notebook_runtime() -> bool:
    try:
        return get_ipython() is not None  # type: ignore[name-defined]
    except NameError:
        return False


EXECUTE_NOTEBOOK = os.environ.get("EXP501_IMPORT_ONLY", "0") != "1" and in_notebook_runtime()

# %% [markdown]
# ## 2. Notebook-safe runtime and path helpers
#
# No helper depends on source-file location state; Jupytext conversion and the
# Kaggle bootstrap therefore preserve config/support-file resolution behavior.


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
    candidates = [Path.cwd() / filename, experiment_dir() / filename]
    for path in candidates:
        if path.exists():
            return path
    matches = sorted(Path.cwd().rglob(filename))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"{filename} did not resolve uniquely: {matches}")


def runtime_output_dir() -> Path:
    path = (
        KAGGLE_WORKING_ROOT / "artifacts"
        if in_notebook_runtime() and KAGGLE_WORKING_ROOT.exists()
        else experiment_dir() / "artifacts"
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


def find_raw_split(split: str) -> Path:
    candidates = [
        project_root() / "data" / "raw" / split,
        Path.cwd() / "data" / "raw" / split,
        KAGGLE_INPUT_ROOT / "competitions" / "rogii-wellbore-geology-prediction" / split,
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
    return [Path.cwd(), project_root(), KAGGLE_INPUT_ROOT, Path("/tmp/kaggle-output")]


CONFIG = read_yaml(find_support_file("config.yaml"))
CONTRACT = read_yaml(find_support_file("candidate_contract.yaml"))
FEATURE_CONTRACT = read_yaml(find_support_file("feature_contract.yaml"))
OUTPUT_DIR = runtime_output_dir()

# %% [markdown]
# ## 3. Authorization, candidate, and compute contract
#
# Implementation approval does not authorize execution. A future Kaggle CPU run
# must separately approve exactly 1 variant, 2 objectives, outer 5, inner 4,
# 40 selector boosters, parent/control retraining 0, HMM/PF/Beam reruns 0, and
# GPU 0. This cell rejects any widened scope before data loading or fitting.

# %%
validate_fixed13_contract(CONTRACT)
candidate_order = candidate_ids(CONTRACT)
compact_names = compact_feature_names(CONTRACT)
execution = CONFIG["execution"]
cost_contract = {
    "active_variants": int(execution["active_variants"]),
    "objectives": int(execution["lightgbm_objectives"]),
    "outer_folds": int(execution["outer_folds"]),
    "inner_folds": int(execution["inner_folds"]),
    "planned_cpu_boosters": int(execution["planned_cpu_boosters"]),
    "parent_control_retraining": bool(execution["parent_control_retraining"]),
    "candidate_hmm_rerun_well_runs": int(execution["candidate_hmm_rerun_well_runs"]),
    "candidate_pf_rerun_well_runs": int(execution["candidate_pf_rerun_well_runs"]),
    "beam_well_runs": int(execution["beam_well_runs"]),
    "gpu_boosters": int(execution["gpu_boosters"]),
    "downstream_tvt_training": bool(execution["downstream_tvt_training"]),
    "inference": bool(execution["inference"]),
    "submission": bool(execution["submission"]),
}
assert candidate_order == [*BASE_CANDIDATE_IDS, ADDED_CANDIDATE_ID]
assert cost_contract == {
    "active_variants": 1,
    "objectives": 2,
    "outer_folds": 5,
    "inner_folds": 4,
    "planned_cpu_boosters": 40,
    "parent_control_retraining": False,
    "candidate_hmm_rerun_well_runs": 0,
    "candidate_pf_rerun_well_runs": 0,
    "beam_well_runs": 0,
    "gpu_boosters": 0,
    "downstream_tvt_training": False,
    "inference": False,
    "submission": False,
}
assert FEATURE_CONTRACT["fold_contract"]["exp490_source_fold_feature_allowed"] is False
assert FEATURE_CONTRACT["fold_contract"]["exp490_source_fold_split_allowed"] is False
print(
    json.dumps(
        {
            "experiment": EXPERIMENT_NAME,
            "route": CONFIG["experiment"]["route"],
            "selector_parent": CONFIG["lineage"]["parent"],
            "candidate_parent": CONFIG["lineage"]["candidate_parent"],
            "candidate_order": candidate_order,
            "primary_candidates": CONTRACT["legal_domains"]["primitive_pair_bank"][
                "candidates"
            ],
            "fixed_fallback_candidates": CONTRACT["legal_domains"]["primitive_fixed_bank"][
                "candidates"
            ],
            "compact_feature_count_from_contract": len(compact_names),
            "execution": cost_contract,
            "run_approved": bool(execution["run_approved"]),
        },
        indent=2,
    )
)

# %% [markdown]
# ## 4. Frozen exp263 / exp490 / exp264 input checks
#
# The exp490 CSV header contains post-freeze truth/error/fold columns, but the
# loader parses only global keys, the HMM prediction, residual-state mean, and
# posterior standard deviation; `id` is reconstructed from `well,row_idx`. The
# saved parent exp264 score and hidden-like assignment resolve here but are
# attached only after fixed13 prediction freeze.

# %%
if EXECUTE_NOTEBOOK:
    if not bool(execution["run_approved"]):
        raise RuntimeError(
            "exp501 Kaggle Stage A/C run is not authorized. Reconfirm "
            "1 variant / 2 objectives / outer5 x inner4 / 40 CPU boosters / "
            "parent-control retraining 0 / HMM-PF-Beam rerun 0 / GPU 0."
        )
    approved_scope = (
        "fixed13_stage_a_plus_stage_c_1_variant_2_objectives_"
        "5_outer_4_inner_40_cpu_boosters_no_control_retraining"
    )
    if execution.get("approved_scope") != approved_scope:
        raise RuntimeError("exp501 approval scope does not match the fixed cost contract")

    started = time.perf_counter()
    roots = search_roots()
    raw_train_dir = find_raw_split("train")
    raw_test_dir = find_raw_split("test")
    cache_root = resolve_exp263_cache_root(CONFIG, roots)

    source_cfg = CONFIG["data"]["exp490_source"]
    prediction_cfg = source_cfg["prediction"]
    prediction_path = resolve_csv_by_payload_sha(
        prediction_cfg["patterns"],
        roots,
        expected_payload_sha256=prediction_cfg["expected_decompressed_sha256"],
        expected_gzip_raw_sha256=prediction_cfg["expected_raw_gzip_sha256"],
        label="saved exp490 Stage 1 full OOF predictions",
    )
    exp490_inputs, exp490_manifest = load_exp490_target_free_inputs(
        prediction_path,
        expected_rows=int(source_cfg["expected_rows"]),
        expected_wells=int(source_cfg["expected_wells"]),
        expected_prediction_gzip_raw_sha256=prediction_cfg["expected_raw_gzip_sha256"],
        expected_prediction_payload_sha256=prediction_cfg["expected_decompressed_sha256"],
    )
    write_json(OUTPUT_DIR / "exp501_exp490_prediction_manifest.json", exp490_manifest)

    parent_cfg = CONFIG["data"]["parent_exp264_stage_c"]
    parent_score_path = resolve_file_by_sha(
        parent_cfg["score_patterns"],
        roots,
        expected_file_sha256=parent_cfg["expected_score_file_sha256"],
        label="corrected exp264 Stage C outer-valid candidate score",
    )
    parent_schema_path = resolve_existing_path(
        CONFIG["data"]["exp251_selected_feature_schema_patterns"], roots
    )
    hidden_like_path = resolve_existing_path(
        CONFIG["data"]["hidden_like_assignment_patterns"], roots
    )
    if sha256_file(hidden_like_path) != str(
        CONFIG["data"]["hidden_like_assignment_expected_sha256"]
    ):
        raise ValueError("hidden-like assignment SHA mismatch")
    print(
        json.dumps(
            {
                "raw_train_dir": str(raw_train_dir),
                "raw_test_dir": str(raw_test_dir),
                "exp263_cache_root": str(cache_root),
                "exp490_target_free_inputs": exp490_manifest,
                "parent_exp264_score": {
                    "path": str(parent_score_path),
                    "sha256": sha256_file(parent_score_path),
                },
                "parent_feature_schema": str(parent_schema_path),
                "hidden_like_assignment": str(hidden_like_path),
            },
            indent=2,
        )
    )

# %% [markdown]
# ## 5. Fixed13 cache and Stage A feature freeze
#
# The unchanged exp263 loader reconstructs the fixed12 bank. exp490 is globally
# joined by `(well, row_idx)`, suffix offset is checked against the selector
# suffix order, and rows are repartitioned only by the exp263 outer fold. Stage A
# performs only truth-free mechanical all-missing/constant/exact-duplicate drops.

# %%
if EXECUTE_NOTEBOOK:
    fixed13_cache = Exp490Fixed13CandidateCache(
        cache_root,
        CONTRACT,
        exp490_inputs=exp490_inputs,
        exp490_manifest=exp490_manifest,
    )

    def fixed13_cache_factory(
        _root: Path, _contract: dict[str, Any]
    ) -> Exp490Fixed13CandidateCache:
        if Path(_root) != Path(cache_root):
            raise ValueError("fixed13 cache root changed after input freeze")
        if candidate_ids(_contract) != candidate_order:
            raise ValueError("fixed13 candidate order changed after input freeze")
        return fixed13_cache

    for fold in range(5):
        preview = fixed13_cache.load_fold(fold)
        assert preview.values.shape[1] == 13
        assert preview.available.all()
        assert preview.candidate_ids == candidate_order
        added_confidence = preview.confidence[ADDED_CANDIDATE_ID]
        assert added_confidence["confidence_valid"].all()
        assert set(EXP490_NATIVE_FIELDS).issubset(added_confidence.columns)

    repartition_manifest = fixed13_cache.selector_repartition_manifest(
        expected_rows=int(CONFIG["validation"]["expected_rows"])
    )
    if not bool(repartition_manifest["passed"]):
        raise RuntimeError(
            "exp490 global-key selector-fold repartition failed: "
            f"{repartition_manifest['checks']}"
        )
    exp490_manifest["selector_fold_repartition"] = repartition_manifest
    write_json(
        OUTPUT_DIR / "exp501_exp490_selector_fold_repartition.json",
        repartition_manifest,
    )
    write_json(OUTPUT_DIR / "exp501_exp490_prediction_manifest.json", exp490_manifest)
    write_exp501_input_contract(
        OUTPUT_DIR / "exp501_input_contract.json",
        config=CONFIG,
        contract=CONTRACT,
        exp490_manifest=exp490_manifest,
        parent_score_path=parent_score_path,
    )
    availability = audit_raw_context_availability(
        raw_train_dir,
        raw_test_dir,
        CONFIG["features"]["raw_context"]["horizontal_numeric_allowlist"],
    )
    availability.to_csv(OUTPUT_DIR / "raw_context_availability_audit.csv", index=False)
    stage_a = run_stage_a(
        config=CONFIG,
        contract=CONTRACT,
        cache_root=cache_root,
        raw_train_dir=raw_train_dir,
        output_dir=OUTPUT_DIR,
        parent_schema_path=parent_schema_path,
        cache_factory=fixed13_cache_factory,
    )
    compact_schema = json.loads((OUTPUT_DIR / "compact_meta_schema.json").read_text())
    if compact_schema["features"] != compact_names:
        raise ValueError("exp501 compact schema differs from the frozen contract")
    print(json.dumps({"stage_a": stage_a}, indent=2))

# %% [markdown]
# ## 6. Stage C strict-nested dual selector
#
# Each outer fold fits four inner models for each unchanged objective. Outer-train
# compact rows use inner OOF scores; outer-valid compact rows use the four-model
# ensemble. The exact run is 40 CPU boosters and 25 compact partitions, with no
# parent control or candidate HMM regeneration.

# %%
if EXECUTE_NOTEBOOK:
    stage_c = run_stage_c(
        config=CONFIG,
        contract=CONTRACT,
        cache_root=cache_root,
        raw_train_dir=raw_train_dir,
        output_dir=OUTPUT_DIR,
        cache_factory=fixed13_cache_factory,
        hard_readout_enabled=True,
    )
    technical_cfg = CONFIG["guards"]["technical"]
    technical_checks = {
        "model_count": int(stage_c["model_count"]) == int(technical_cfg["expected_models"]),
        "compact_partition_count": int(stage_c["compact_partition_count"])
        == int(technical_cfg["expected_compact_partitions"]),
        "compact_rows": int(stage_c["compact_rows"])
        == int(technical_cfg["expected_compact_rows"]),
        "outer_valid_score_long_rows": int(stage_c["outer_valid_score_long_rows"])
        == int(technical_cfg["expected_outer_valid_score_long_rows"]),
        "leakage_audit": bool(stage_c["leakage_audit"]["passed"]),
        "exp490_forbidden_columns_loaded_before_feature_freeze": int(
            exp490_manifest[
                "forbidden_truth_error_role_episode_fold_scope_gate_columns_loaded"
            ]
        )
        == 0,
        "exp490_global_key_suffix_and_selector_fold_repartition": bool(
            repartition_manifest["passed"]
        ),
        "exp490_source_fold_not_loaded_or_used": (
            not bool(exp490_manifest["upstream_source_fold_column_loaded"])
            and not bool(exp490_manifest["upstream_source_fold_used_as_model_feature"])
        ),
        "exp490_candidate_and_native_confidence_finite": float(
            exp490_manifest["candidate_and_native_confidence_finite_fraction"]
        )
        == 1.0,
        "exp490_raw_and_decompressed_sha": (
            exp490_manifest["prediction_file_sha256"]
            == prediction_cfg["expected_raw_gzip_sha256"]
            and exp490_manifest["prediction_payload_sha256"]
            == prediction_cfg["expected_decompressed_sha256"]
        ),
    }
    if not all(technical_checks.values()):
        raise RuntimeError(f"exp501 technical checks failed: {technical_checks}")
    print(json.dumps({"stage_c": stage_c, "technical_checks": technical_checks}, indent=2))

# %% [markdown]
# ## 7. Fixed13 versus fixed12 scientific and reranking readouts
#
# Parent scores, truth-bearing scopes, candidate usage, H512/whole-well oracle,
# and incumbent reranking are attached only after all 40 models and hard choices
# freeze. The preregistered pooled/fold/scope/by-well AND gate cannot be rescued
# by these diagnostic readouts.

# %%
if EXECUTE_NOTEBOOK:
    new_score_path = OUTPUT_DIR / "nested_outer_valid_candidate_score.parquet"
    paired = pair_selector_scores(
        new_score_path=new_score_path,
        parent_score_path=parent_score_path,
        contract=CONTRACT,
    )
    gate, by_well = build_fixed13_integration_readout(
        paired=paired,
        hidden_like_assignment_path=hidden_like_path,
        raw_train_dir=raw_train_dir,
        score_summary=stage_c,
        guard_config=CONFIG["guards"]["integration"],
        output_dir=OUTPUT_DIR,
    )
    scope_metrics = pd.read_csv(OUTPUT_DIR / "exp501_fixed13_vs_fixed12_scope_metrics.csv")
    pooled = scope_metrics.loc[scope_metrics["scope"].eq("pooled")].iloc[0]
    if (
        abs(
            float(pooled["parent_fixed12_hard_rmse"])
            - float(parent_cfg["expected_hard_primary_oof_rmse"])
        )
        > 1.0e-6
    ):
        raise ValueError("parent fixed12 hard-primary RMSE parity failed")
    if (
        abs(
            float(pooled["fixed_fallback_rmse"])
            - float(parent_cfg["expected_fixed_fallback_oof_rmse"])
        )
        > 1.0e-6
    ):
        raise ValueError("fixed fallback RMSE parity failed")

    diagnostic_cfg = CONFIG["guards"]["diagnostic_only"]
    novelty = build_postfreeze_addone_novelty_readout(
        new_score_path=new_score_path,
        output_dir=OUTPUT_DIR,
    )
    reranking = build_incumbent_reranking_diagnostic(
        new_score_path=new_score_path,
        paired=paired,
        by_well=by_well,
        contract=CONTRACT,
        output_dir=OUTPUT_DIR,
        quantile_bins=int(diagnostic_cfg["diagnostic_quantile_bins"]),
    )
    print(
        json.dumps(
            {
                "scientific_gate": gate,
                "postfreeze_addone_novelty": novelty,
                "incumbent_reranking": reranking,
            },
            indent=2,
        )
    )

# %% [markdown]
# ## 8. Feature importance and generated artifacts
#
# Mean fold/inner-model feature importance is saved and plotted by objective.
# This is report-only; it never selects a post-hoc feature subset.

# %%
if EXECUTE_NOTEBOOK:
    if plt is None:
        raise ModuleNotFoundError("matplotlib is required for the feature-importance plot")
    importance = pd.read_csv(
        OUTPUT_DIR / "nested_feature_importance_by_objective_outer_inner.csv"
    )
    importance_summary = (
        importance.groupby(["objective", "feature", "importance_type"], sort=True)[
            "importance"
        ]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": "importance_mean",
                "std": "importance_std",
                "count": "model_count",
            }
        )
    )
    importance_summary_path = OUTPUT_DIR / "exp501_feature_importance_summary.csv"
    importance_summary.to_csv(importance_summary_path, index=False)
    gain = importance_summary.loc[importance_summary["importance_type"].eq("gain")].copy()
    gain["rank"] = gain.groupby("objective")["importance_mean"].rank(
        method="first", ascending=False
    )
    top_gain = gain.loc[gain["rank"].le(20)].sort_values(
        ["objective", "importance_mean"], ascending=[True, True]
    )
    objectives = sorted(top_gain["objective"].unique().tolist())
    fig, axes = plt.subplots(1, len(objectives), figsize=(8 * len(objectives), 8))
    if len(objectives) == 1:
        axes = [axes]
    for axis, objective in zip(axes, objectives, strict=True):
        selected = top_gain.loc[top_gain["objective"].eq(objective)]
        axis.barh(selected["feature"], selected["importance_mean"])
        axis.set_title(f"{objective}: mean gain")
        axis.set_xlabel("mean gain across outer/inner models")
    fig.tight_layout()
    importance_plot_path = OUTPUT_DIR / "exp501_feature_importance_top20.png"
    fig.savefig(importance_plot_path, dpi=140, bbox_inches="tight")
    plt.show()
    print(top_gain[["objective", "rank", "feature", "importance_mean"]])

# %% [markdown]
# ## 9. Reproducibility summary and fixed stop
#
# A failed gate closes the branch without same-OOF rescue. A pass records only
# train-side selector evidence; current-test generation, downstream TVT,
# inference, and submission remain outside this implementation and approval.

# %%
if EXECUTE_NOTEBOOK:
    decision = str(gate["decision"])
    summary = {
        "status": "kaggle_cpu_stage_a_stage_c_completed",
        "decision": decision,
        "scientific_gate_passed": bool(gate["passed"]),
        "execution": cost_contract,
        "rows": int(CONFIG["validation"]["expected_rows"]),
        "wells": int(CONFIG["validation"]["expected_wells"]),
        "candidate_count": len(candidate_order),
        "compact_feature_count": len(compact_names),
        "stage_a": stage_a,
        "stage_c": stage_c,
        "technical_checks": technical_checks,
        "integration_gate": gate,
        "postfreeze_addone_novelty": novelty,
        "incumbent_reranking": reranking,
        "exp490_target_free_inputs": exp490_manifest,
        "exp490_selector_fold_repartition": repartition_manifest,
        "parent_exp264_score_sha256": sha256_file(parent_score_path),
        "feature_importance_summary_sha256": sha256_file(importance_summary_path),
        "feature_importance_plot_sha256": sha256_file(importance_plot_path),
        "elapsed_seconds": time.perf_counter() - started,
        "kaggle_runtime_env": {
            key: os.environ.get(key)
            for key in (
                "KAGGLE_KERNEL_RUN_TYPE",
                "KAGGLE_KERNEL_INFERENCE_RUN_ID",
                "KAGGLE_KERNEL_INFERENCE_RUN_TYPE",
            )
        },
        "deterministic_submission_anchor": False,
        "downstream_tvt_training": False,
        "inference": False,
        "submission": False,
    }
    summary_path = OUTPUT_DIR / "exp501_summary.json"
    write_json(summary_path, summary)
    reproducibility_path = OUTPUT_DIR / "reproducibility_manifest.json"
    reproducibility = json.loads(reproducibility_path.read_text())
    reproducibility.update(
        {
            "exp501_status": summary["status"],
            "decision": decision,
            "deterministic_submission_anchor": False,
            "exp490_target_free_inputs": exp490_manifest,
            "exp490_selector_fold_repartition": repartition_manifest,
            "parent_exp264_score_sha256": summary["parent_exp264_score_sha256"],
            "scientific_gate": gate,
            "postfreeze_addone_novelty": novelty,
            "incumbent_reranking": reranking,
            "exp501_summary_sha256": sha256_file(summary_path),
        }
    )
    write_json(reproducibility_path, reproducibility)
    if KAGGLE_WORKING_ROOT.exists():
        write_json(KAGGLE_WORKING_ROOT / "metrics.json", summary)
    print("FINAL_SUMMARY", json.dumps(summary, sort_keys=True))
