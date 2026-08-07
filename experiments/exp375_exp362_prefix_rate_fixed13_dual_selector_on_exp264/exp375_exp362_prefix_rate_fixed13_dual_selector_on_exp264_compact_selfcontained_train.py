# %% [markdown]
# # exp375 exp362 prefix-rate fixed13 dual selector on exp264 — train
#
# Add the saved exp362 prefix-rate-only exact-HMM OOF as exactly one new
# candidate to the corrected exp264 candidate-long dual selector. Candidate
# posterior standard deviation and well-normalized HMM likelihood are retained
# as native confidence; donor-slope fields are excluded. Stage A refreezes the
# raw-test-safe selector schema, then Stage C trains the unchanged nested dual
# selector. The saved fixed12 score and add-one oracle diagnostic are read only
# after selector prediction freeze. Downstream TVT, inference, and submission
# are excluded.

# %% [markdown]
# ## Contents
# 1. Imports and immutable experiment boundary
# 2. Runtime, path, and serialization helpers
# 3. Candidate, compute, and leakage contract
# 4. Input resolution and SHA checks
# 5. Fixed13 cache and Stage A feature freeze
# 6. Stage C nested selector training
# 7. Fixed13 versus fixed12 integration readout
# 8. Metrics and generated artifacts

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
from src.exp362_fixed13_candidate_cache import (
    ADDED_CANDIDATE_ID,
    BASE_CANDIDATE_IDS,
    Exp362Fixed13CandidateCache,
    build_postfreeze_addone_novelty_readout,
    build_fixed13_integration_readout,
    load_exp362_oof,
    resolve_file_by_sha,
    resolve_parent_score_file,
    validate_fixed13_contract,
    write_exp375_input_contract,
)

EXPERIMENT_NAME = "exp375_exp362_prefix_rate_fixed13_dual_selector_on_exp264"
OUTPUT_PREFIX = EXPERIMENT_NAME
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")


def in_notebook_runtime() -> bool:
    try:
        return get_ipython() is not None  # type: ignore[name-defined]
    except NameError:
        return False


EXECUTE_NOTEBOOK = (
    os.environ.get("EXP375_IMPORT_ONLY", "0") != "1" and in_notebook_runtime()
)

# %% [markdown]
# ## 2. Runtime, path, and serialization helpers

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
    ]
    for path in candidates:
        if path.exists():
            return path
    matches = sorted(Path.cwd().rglob(filename))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"{filename} did not resolve uniquely: {matches}")


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
# ## 3. Candidate, compute, and leakage contract
#
# The cost is fixed before any model fit: one fixed13 variant, two objectives,
# five downstream outer folds, four inner folds, and 40 CPU boosters. The
# corrected exp264 fixed12 selector is a saved comparison and is not retrained.

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
    "gpu_boosters": int(execution["gpu_boosters"]),
    "downstream_tvt_training": bool(execution["downstream_tvt_training"]),
    "inference": bool(execution["inference"]),
    "submission": bool(execution["submission"]),
}
assert candidate_order == [*BASE_CANDIDATE_IDS, ADDED_CANDIDATE_ID]
assert len(candidate_order) == 13
assert len(compact_names) == int(
    CONFIG["guards"]["technical"]["expected_compact_feature_count"]
)
assert cost_contract == {
    "active_variants": 1,
    "objectives": 2,
    "outer_folds": 5,
    "inner_folds": 4,
    "planned_cpu_boosters": 40,
    "parent_control_retraining": False,
    "gpu_boosters": 0,
    "downstream_tvt_training": False,
    "inference": False,
    "submission": False,
}
print(
    json.dumps(
        {
            "experiment": EXPERIMENT_NAME,
            "route": CONFIG["experiment"]["route"],
            "candidate_order": candidate_order,
            "compact_feature_count": len(compact_names),
            "execution": cost_contract,
            "run_approved": bool(execution["run_approved"]),
        },
        indent=2,
    )
)

# %% [markdown]
# ## 4. Input resolution and SHA checks
#
# Exp362 is opened through the six target-free allowlisted columns only. Donor
# schedule, mu_rate, truth, and evaluation columns are deliberately not read.
# The parent exp264 score is a post-fit audit input and never enters Stage A or
# selector fitting.

# %%
if EXECUTE_NOTEBOOK:
    if not bool(execution["run_approved"]):
        raise RuntimeError(
            "exp375 Kaggle train is not authorized. Set execution.run_approved "
            "only after confirming 1 variant / 2 objectives / outer5 x inner4 / "
            "40 CPU boosters / parent-control retraining 0."
        )
    if execution.get("approved_scope") != (
        "fixed13_stage_a_plus_stage_c_1_variant_2_objectives_"
        "5_outer_4_inner_40_cpu_boosters_no_control_retraining"
    ):
        raise RuntimeError("exp375 approval scope does not match the fixed cost contract")

    started = time.perf_counter()
    roots = search_roots()
    raw_train_dir = find_raw_split("train")
    raw_test_dir = find_raw_split("test")
    cache_root = resolve_exp263_cache_root(CONFIG, roots)

    exp362_cfg = CONFIG["data"]["exp362_oof"]
    exp362_path = resolve_file_by_sha(
        exp362_cfg["patterns"],
        roots,
        expected_file_sha256=exp362_cfg.get("expected_file_sha256"),
        expected_decompressed_sha256=exp362_cfg["expected_decompressed_sha256"],
        label="saved exp362 prefix-rate exact-HMM OOF",
    )
    exp362_oof, exp362_manifest = load_exp362_oof(
        exp362_path,
        expected_rows=int(CONFIG["validation"]["expected_rows"]),
        expected_wells=int(CONFIG["validation"]["expected_wells"]),
        expected_file_sha256=exp362_cfg.get("expected_file_sha256"),
        expected_decompressed_sha256=exp362_cfg["expected_decompressed_sha256"],
        expected_prediction_logical_sha256=exp362_cfg[
            "expected_prediction_logical_sha256"
        ],
    )
    write_json(OUTPUT_DIR / "exp375_exp362_oof_manifest.json", exp362_manifest)

    parent_cfg = CONFIG["data"]["parent_exp264_stage_c"]
    parent_score_path = resolve_parent_score_file(
        parent_cfg["score_patterns"],
        roots,
        parent_cfg["expected_score_file_sha256"],
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
                "exp362_oof": exp362_manifest,
                "parent_exp264_score": {
                    "path": str(parent_score_path),
                    "sha256": sha256_file(parent_score_path),
                },
                "parent_schema": str(parent_schema_path),
                "hidden_like_assignment": str(hidden_like_path),
            },
            indent=2,
        )
    )

# %% [markdown]
# ## 5. Fixed13 cache and Stage A feature freeze
#
# The exp263 loader reconstructs the original 12 candidates. The extension
# globally joins exp362 by `(well_id, row_idx)`, preserves its source fold as
# provenance, and repartitions rows by the exp263 selector fold. Source fold is
# never a model feature. Candidate posterior std and per-well normalized HMM
# likelihood are the only added native-confidence fields. Stage A then
# refreezes the candidate-long schema with the corrected exp264 audit policy.

# %%
if EXECUTE_NOTEBOOK:
    fixed13_cache = Exp362Fixed13CandidateCache(
        cache_root,
        CONTRACT,
        exp362_oof=exp362_oof,
        exp362_manifest=exp362_manifest,
    )

    def fixed13_cache_factory(
        _root: Path, _contract: dict[str, Any]
    ) -> Exp362Fixed13CandidateCache:
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
        assert {
            "sigma_tvt",
            "source_loglik",
            "loglik_per_row",
        }.issubset(added_confidence.columns)

    exp362_repartition_manifest = fixed13_cache.selector_repartition_manifest(
        expected_rows=int(CONFIG["validation"]["expected_rows"])
    )
    if not bool(exp362_repartition_manifest["passed"]):
        raise RuntimeError(
            "exp362 global-key selector-fold repartition audit failed: "
            f"{exp362_repartition_manifest['checks']}"
        )
    exp362_manifest["selector_fold_repartition"] = exp362_repartition_manifest
    write_json(
        OUTPUT_DIR / "exp375_exp362_selector_fold_repartition.json",
        exp362_repartition_manifest,
    )
    write_json(OUTPUT_DIR / "exp375_exp362_oof_manifest.json", exp362_manifest)

    write_exp375_input_contract(
        OUTPUT_DIR / "exp375_input_contract.json",
        config=CONFIG,
        contract=CONTRACT,
        exp362_manifest=exp362_manifest,
        parent_score_path=parent_score_path,
    )
    availability = audit_raw_context_availability(
        raw_train_dir,
        raw_test_dir,
        CONFIG["features"]["raw_context"]["horizontal_numeric_allowlist"],
    )
    availability.to_csv(
        OUTPUT_DIR / "raw_context_availability_audit.csv", index=False
    )
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
    if len(compact_schema["features"]) != 77:
        raise ValueError("fixed13 compact feature count must be 77")
    if compact_schema["features"] != compact_names:
        raise ValueError("fixed13 compact schema differs from the frozen contract")
    print(json.dumps({"stage_a": stage_a}, indent=2))

# %% [markdown]
# ## 6. Stage C nested selector training
#
# For every downstream outer fold, four inner models are fit for each of the two
# objectives. Outer-train compact rows are inner OOF; outer-valid rows are the
# four-model ensemble. This produces 40 saved CPU boosters and 25 partitions.

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
        "model_count": int(stage_c["model_count"])
        == int(technical_cfg["expected_models"]),
        "compact_partition_count": int(stage_c["compact_partition_count"])
        == int(technical_cfg["expected_compact_partitions"]),
        "compact_rows": int(stage_c["compact_rows"])
        == int(technical_cfg["expected_compact_rows"]),
        "outer_valid_score_long_rows": int(stage_c["outer_valid_score_long_rows"])
        == int(technical_cfg["expected_outer_valid_score_long_rows"]),
        "score_guard": bool(stage_c["score_guard"]["passed"]),
        "leakage_audit": bool(stage_c["leakage_audit"]["passed"]),
        "exp362_truth_or_error_columns_loaded": int(
            exp362_manifest["truth_or_error_columns_loaded"]
        )
        == 0,
        "exp362_global_key_selector_fold_repartition": bool(
            exp362_repartition_manifest["passed"]
        ),
        "exp362_source_fold_not_used_as_model_feature": not bool(
            exp362_repartition_manifest["source_fold_used_as_model_feature"]
        ),
        "exp362_native_confidence_finite": float(
            exp362_manifest["native_confidence_finite_fraction"]
        )
        == 1.0,
    }
    if not all(technical_checks.values()):
        raise RuntimeError(f"exp375 technical checks failed: {technical_checks}")
    print(json.dumps({"stage_c": stage_c, "technical_checks": technical_checks}, indent=2))

# %% [markdown]
# ## 7. Fixed13 versus fixed12 integration readout
#
# The saved corrected exp264 outer-valid score is paired by row with the new
# fixed13 score. This reports candidate usage and RMSE deltas on pooled, fold,
# near, 1000+, hidden-like, and by-well surfaces. It is evaluation only and is
# never fed back into training.

# %%
if EXECUTE_NOTEBOOK:
    new_score_path = OUTPUT_DIR / "nested_outer_valid_candidate_score.parquet"
    gate = build_fixed13_integration_readout(
        new_score_path=new_score_path,
        parent_score_path=parent_score_path,
        hidden_like_assignment_path=hidden_like_path,
        contract=CONTRACT,
        score_summary=stage_c,
        guard_config=CONFIG["guards"]["integration"],
        output_dir=OUTPUT_DIR,
    )
    scope_metrics = pd.read_csv(
        OUTPUT_DIR / "exp375_fixed13_vs_fixed12_scope_metrics.csv"
    )
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
    print(json.dumps({"integration_gate": gate}, indent=2))

    novelty = build_postfreeze_addone_novelty_readout(
        new_score_path=new_score_path,
        output_dir=OUTPUT_DIR,
        tie_atol_squared_ft=float(
            CONFIG["guards"]["diagnostic_only"]["postfreeze_addone_novelty"][
                "tie_atol_squared_ft"
            ]
        ),
    )
    print(json.dumps({"postfreeze_addone_novelty": novelty}, indent=2))

# %% [markdown]
# ## 8. Metrics and generated artifacts
#
# A failed scientific gate closes only downstream promotion; the Kaggle audit
# itself remains complete so the negative result and all SHA evidence persist.

# %%
if EXECUTE_NOTEBOOK:
    decision = (
        "PASS_FIXED13_SELECTOR_INTEGRATION"
        if bool(gate["passed"])
        else "FAIL_CLOSE_FIXED13_SELECTOR_BRANCH"
    )
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
        "exp362_oof": exp362_manifest,
        "exp362_selector_fold_repartition": exp362_repartition_manifest,
        "parent_exp264_score_sha256": sha256_file(parent_score_path),
        "elapsed_seconds": time.perf_counter() - started,
        "downstream_tvt_training": False,
        "inference": False,
        "submission": False,
    }
    write_json(OUTPUT_DIR / "exp375_summary.json", summary)
    reproducibility_path = OUTPUT_DIR / "reproducibility_manifest.json"
    reproducibility = json.loads(reproducibility_path.read_text())
    reproducibility.update(
        {
            "exp375_status": summary["status"],
            "decision": decision,
            "exp362_oof": exp362_manifest,
            "exp362_selector_fold_repartition": exp362_repartition_manifest,
            "parent_exp264_score_sha256": summary[
                "parent_exp264_score_sha256"
            ],
            "scientific_gate": gate,
            "postfreeze_addone_novelty": novelty,
            "exp375_summary_sha256": sha256_file(
                OUTPUT_DIR / "exp375_summary.json"
            ),
        }
    )
    write_json(reproducibility_path, reproducibility)
    if KAGGLE_WORKING_ROOT.exists():
        write_json(KAGGLE_WORKING_ROOT / "metrics.json", summary)
    print("FINAL_SUMMARY", json.dumps(summary, sort_keys=True))
