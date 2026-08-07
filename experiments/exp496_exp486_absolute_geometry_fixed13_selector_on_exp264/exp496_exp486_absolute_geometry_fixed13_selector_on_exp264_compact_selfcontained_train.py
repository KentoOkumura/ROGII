# %% [markdown]
# # exp496 exp486 absolute-geometry fixed13 selector on exp264 — train
#
# Add the saved exp486 absolute-geometry likelihood-PF prediction as exactly
# one new candidate to the corrected exp264 dual selector. The exp486
# prediction and absolute mechanism ledger are SHA-frozen target-free inputs;
# no PF, HMM, Beam, parent selector, downstream TVT, inference, or submission
# work is performed. Stage A refreezes the candidate-long schema, Stage C fits
# the unchanged outer-5 / inner-4 dual selector, and every scientific or
# reranking readout is computed only after selector prediction freeze.

# %% [markdown]
# ## Contents
# 1. Imports and immutable experiment boundary
# 2. Notebook-safe runtime and path helpers
# 3. Authorization, candidate, and compute contract
# 4. Frozen exp263 / exp486 / exp264 input checks
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
from src.exp486_fixed13_candidate_cache import (
    ADDED_CANDIDATE_ID,
    BASE_CANDIDATE_IDS,
    EXP486_LEDGER_FIELDS,
    Exp486Fixed13CandidateCache,
    build_fixed13_integration_readout,
    build_incumbent_reranking_diagnostic,
    build_postfreeze_addone_novelty_readout,
    load_exp486_target_free_inputs,
    pair_selector_scores,
    resolve_csv_by_payload_sha,
    resolve_file_by_sha,
    validate_fixed13_contract,
    write_exp496_input_contract,
)

EXPERIMENT_NAME = "exp496_exp486_absolute_geometry_fixed13_selector_on_exp264"
OUTPUT_PREFIX = EXPERIMENT_NAME
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")


def in_notebook_runtime() -> bool:
    try:
        return get_ipython() is not None  # type: ignore[name-defined]
    except NameError:
        return False


EXECUTE_NOTEBOOK = os.environ.get("EXP496_IMPORT_ONLY", "0") != "1" and in_notebook_runtime()

# %% [markdown]
# ## 2. Notebook-safe runtime and path helpers
#
# The Jupytext source must work after conversion, so no helper below depends on
# source-file location state. The package bootstrap restores `src/`, config,
# and contracts under the current working directory on Kaggle.


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
    return [
        Path.cwd(),
        project_root(),
        KAGGLE_INPUT_ROOT,
        Path("/tmp/kaggle-output"),
    ]


CONFIG = read_yaml(find_support_file("config.yaml"))
CONTRACT = read_yaml(find_support_file("candidate_contract.yaml"))
FEATURE_CONTRACT = read_yaml(find_support_file("feature_contract.yaml"))
OUTPUT_DIR = runtime_output_dir()

# %% [markdown]
# ## 3. Authorization, candidate, and compute contract
#
# The user approved this exact Kaggle CPU run on 2026-07-31: 1 variant,
# 2 objectives, outer 5, inner 4, 40 selector boosters, parent/control
# retraining 0, PF/HMM/Beam reruns 0, and GPU 0. The exact scope string below
# prevents that authorization from expanding silently.

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
    "candidate_pf_rerun_well_runs": int(execution["candidate_pf_rerun_well_runs"]),
    "hmm_well_runs": int(execution["hmm_well_runs"]),
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
    "candidate_pf_rerun_well_runs": 0,
    "hmm_well_runs": 0,
    "beam_well_runs": 0,
    "gpu_boosters": 0,
    "downstream_tvt_training": False,
    "inference": False,
    "submission": False,
}
assert FEATURE_CONTRACT["fold_contract"]["upstream_exp226_fold_feature_allowed"] is False
print(
    json.dumps(
        {
            "experiment": EXPERIMENT_NAME,
            "route": CONFIG["experiment"]["route"],
            "selector_parent": CONFIG["lineage"]["parent"],
            "candidate_parent": CONFIG["lineage"]["candidate_parent"],
            "candidate_order": candidate_order,
            "primary_candidates": CONTRACT["legal_domains"]["primitive_pair_bank"]["candidates"],
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
# ## 4. Frozen exp263 / exp486 / exp264 input checks
#
# Only the absolute prediction and the five absolute-mechanism confidence
# fields are parsed from exp486. Residual prediction/ledger, truth, controls,
# upstream folds, roles, scopes, by-well metrics, and gates are not opened.
# The saved exp264 score and hidden-like assignment are resolved now, but are
# not attached until every new selector score and choice has frozen.

# %%
if EXECUTE_NOTEBOOK:
    if not bool(execution["run_approved"]):
        raise RuntimeError(
            "exp496 Kaggle Stage A/C run is not authorized. Reconfirm "
            "1 variant / 2 objectives / outer5 x inner4 / 40 CPU boosters / "
            "parent-control retraining 0 / PF-HMM-Beam rerun 0 / GPU 0, then "
            "set the exact approved scope."
        )
    approved_scope = (
        "fixed13_stage_a_plus_stage_c_1_variant_2_objectives_"
        "5_outer_4_inner_40_cpu_boosters_no_control_retraining"
    )
    if execution.get("approved_scope") != approved_scope:
        raise RuntimeError("exp496 approval scope does not match the fixed cost contract")

    started = time.perf_counter()
    roots = search_roots()
    raw_train_dir = find_raw_split("train")
    raw_test_dir = find_raw_split("test")
    cache_root = resolve_exp263_cache_root(CONFIG, roots)

    source_cfg = CONFIG["data"]["exp486_source"]
    prediction_cfg = source_cfg["prediction"]
    ledger_cfg = source_cfg["absolute_ledger"]
    freeze_cfg = source_cfg["freeze_manifest"]
    prediction_path = resolve_csv_by_payload_sha(
        prediction_cfg["patterns"],
        roots,
        expected_payload_sha256=prediction_cfg["expected_decompressed_payload_sha256"],
        expected_gzip_raw_sha256=prediction_cfg["expected_gzip_raw_sha256"],
        label="saved exp486 Stage 1 predictions",
    )
    ledger_path = resolve_csv_by_payload_sha(
        ledger_cfg["patterns"],
        roots,
        expected_payload_sha256=ledger_cfg["expected_decompressed_payload_sha256"],
        expected_gzip_raw_sha256=ledger_cfg["expected_gzip_raw_sha256"],
        label="saved exp486 Stage 1 absolute mechanism ledger",
    )
    freeze_path = resolve_file_by_sha(
        freeze_cfg["patterns"],
        roots,
        expected_file_sha256=freeze_cfg["expected_raw_sha256"],
        label="saved exp486 Stage 1 freeze manifest",
    )
    exp486_inputs, exp486_manifest = load_exp486_target_free_inputs(
        prediction_path,
        ledger_path,
        freeze_path,
        expected_rows=int(source_cfg["expected_rows"]),
        expected_wells=int(source_cfg["expected_wells"]),
        expected_prediction_gzip_raw_sha256=prediction_cfg["expected_gzip_raw_sha256"],
        expected_prediction_payload_sha256=prediction_cfg["expected_decompressed_payload_sha256"],
        expected_prediction_upstream_logical_sha256=prediction_cfg[
            "expected_upstream_logical_sha256"
        ],
        expected_ledger_gzip_raw_sha256=ledger_cfg["expected_gzip_raw_sha256"],
        expected_ledger_payload_sha256=ledger_cfg["expected_decompressed_payload_sha256"],
        expected_freeze_manifest_sha256=freeze_cfg["expected_raw_sha256"],
        expected_scientific_contract_sha256=source_cfg["expected_scientific_contract_sha256"],
        expected_exp226_geometry_decompressed_sha256=source_cfg[
            "expected_exp226_geometry_decompressed_sha256"
        ],
    )
    write_json(OUTPUT_DIR / "exp496_exp486_prediction_manifest.json", exp486_manifest)

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
                "exp486_target_free_inputs": exp486_manifest,
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
# The exp263 loader reconstructs the unchanged 12-candidate score inventory.
# exp486 is globally joined by `(well_id, row_idx)` and then repartitioned by
# the exp263 selector outer fold. The upstream exp226 fold remains a safety
# audit property and is never loaded or encoded. Stage A sees target-free
# candidate values, raw-test-safe context, disagreement, and confidence only;
# all-missing, constant, and exact-duplicate drops happen before truth access.

# %%
if EXECUTE_NOTEBOOK:
    fixed13_cache = Exp486Fixed13CandidateCache(
        cache_root,
        CONTRACT,
        exp486_inputs=exp486_inputs,
        exp486_manifest=exp486_manifest,
    )

    def fixed13_cache_factory(
        _root: Path, _contract: dict[str, Any]
    ) -> Exp486Fixed13CandidateCache:
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
        assert set(EXP486_LEDGER_FIELDS).issubset(added_confidence.columns)

    repartition_manifest = fixed13_cache.selector_repartition_manifest(
        expected_rows=int(CONFIG["validation"]["expected_rows"])
    )
    if not bool(repartition_manifest["passed"]):
        raise RuntimeError(
            f"exp486 global-key selector-fold repartition failed: {repartition_manifest['checks']}"
        )
    exp486_manifest["selector_fold_repartition"] = repartition_manifest
    write_json(
        OUTPUT_DIR / "exp496_exp486_selector_fold_repartition.json",
        repartition_manifest,
    )
    write_json(OUTPUT_DIR / "exp496_exp486_prediction_manifest.json", exp486_manifest)
    write_exp496_input_contract(
        OUTPUT_DIR / "exp496_input_contract.json",
        config=CONFIG,
        contract=CONTRACT,
        exp486_manifest=exp486_manifest,
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
        raise ValueError("exp496 compact schema differs from the frozen contract")
    print(json.dumps({"stage_a": stage_a}, indent=2))

# %% [markdown]
# ## 6. Stage C strict-nested dual selector
#
# For every downstream outer fold, four inner models are fitted for each of the
# two unchanged objectives. Outer-train compact rows use inner OOF scores;
# outer-valid compact rows use the four inner-model ensemble. The run therefore
# creates 40 CPU selector boosters and 25 compact partitions, with no parent
# control retraining.

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
        "compact_rows": int(stage_c["compact_rows"]) == int(technical_cfg["expected_compact_rows"]),
        "outer_valid_score_long_rows": int(stage_c["outer_valid_score_long_rows"])
        == int(technical_cfg["expected_outer_valid_score_long_rows"]),
        "leakage_audit": bool(stage_c["leakage_audit"]["passed"]),
        "exp486_forbidden_columns_loaded_before_feature_freeze": int(
            exp486_manifest["forbidden_truth_error_control_role_fold_scope_gate_columns_loaded"]
        )
        == 0,
        "exp486_global_key_selector_fold_repartition": bool(repartition_manifest["passed"]),
        "exp486_upstream_fold_not_loaded_or_used": (
            not bool(exp486_manifest["upstream_fold_column_loaded"])
            and not bool(exp486_manifest["upstream_fold_used_as_model_feature"])
        ),
        "exp486_candidate_and_native_confidence_finite": float(
            exp486_manifest["candidate_and_native_confidence_finite_fraction"]
        )
        == 1.0,
        "exp486_payload_and_freeze_manifest_sha": all(
            exp486_manifest["freeze_manifest_checks"].values()
        ),
    }
    if not all(technical_checks.values()):
        raise RuntimeError(f"exp496 technical checks failed: {technical_checks}")
    print(json.dumps({"stage_c": stage_c, "technical_checks": technical_checks}, indent=2))

# %% [markdown]
# ## 7. Fixed13 versus fixed12 scientific and reranking readouts
#
# Only after all 40 models, scores, and hard choices freeze do we pair the new
# score with the saved corrected exp264 score, attach GR/scope context, and
# evaluate the preregistered all-AND gate. H512/whole-well oracle headroom and
# incumbent reranking by score margin/entropy are diagnostic only and cannot
# rescue a failed gate.

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
    scope_metrics = pd.read_csv(OUTPUT_DIR / "exp496_fixed13_vs_fixed12_scope_metrics.csv")
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
# Fold/inner-model feature importance is averaged by objective. The table and
# plot make the added mechanism fields and disagreement surface auditable; they
# are not used to select a post-hoc feature subset.

# %%
if EXECUTE_NOTEBOOK:
    if plt is None:
        raise ModuleNotFoundError("matplotlib is required for the feature-importance plot")
    importance = pd.read_csv(OUTPUT_DIR / "nested_feature_importance_by_objective_outer_inner.csv")
    importance_summary = (
        importance.groupby(["objective", "feature", "importance_type"], sort=True)["importance"]
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
    importance_summary_path = OUTPUT_DIR / "exp496_feature_importance_summary.csv"
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
    importance_plot_path = OUTPUT_DIR / "exp496_feature_importance_top20.png"
    fig.savefig(importance_plot_path, dpi=140, bbox_inches="tight")
    plt.show()
    print(top_gain[["objective", "rank", "feature", "importance_mean"]])

# %% [markdown]
# ## 9. Reproducibility summary and fixed stop
#
# A failed scientific gate closes this branch under the fixed decision. A pass
# records train-side evidence only: downstream TVT, current-test exp486
# generation, inference, and submission still require new design and approval.

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
        "exp486_target_free_inputs": exp486_manifest,
        "exp486_selector_fold_repartition": repartition_manifest,
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
    summary_path = OUTPUT_DIR / "exp496_summary.json"
    write_json(summary_path, summary)
    reproducibility_path = OUTPUT_DIR / "reproducibility_manifest.json"
    reproducibility = json.loads(reproducibility_path.read_text())
    reproducibility.update(
        {
            "exp496_status": summary["status"],
            "decision": decision,
            "deterministic_submission_anchor": False,
            "exp486_target_free_inputs": exp486_manifest,
            "exp486_selector_fold_repartition": repartition_manifest,
            "parent_exp264_score_sha256": summary["parent_exp264_score_sha256"],
            "scientific_gate": gate,
            "postfreeze_addone_novelty": novelty,
            "incumbent_reranking": reranking,
            "exp496_summary_sha256": sha256_file(summary_path),
        }
    )
    write_json(reproducibility_path, reproducibility)
    if KAGGLE_WORKING_ROOT.exists():
        write_json(KAGGLE_WORKING_ROOT / "metrics.json", summary)
    print("FINAL_SUMMARY", json.dumps(summary, sort_keys=True))
