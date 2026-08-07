from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.candidate_selector_pipeline import (
    candidate_ids,
    sha256_file,
    write_json,
)
from src.candidate_task_weighting import validate_inverse_rmse_weight_config


EXPERIMENT_NAME = "exp407_fold_safe_inverse_rmse_weighted_dual_selector_on_exp264"
OBJECTIVES = ["pred_abs_error", "p_within10"]
HIDDEN_LIKE_SCOPES = {
    "hidden_like_spatial": "verification_like_spatial_role",
    "hidden_like_typewell_purged": "verification_like_typewell_purged_role",
}


def resolve_pinned_input(
    patterns: Sequence[str],
    search_roots: Sequence[Path],
    *,
    expected_sha256: str,
    label: str,
) -> Path:
    matches: set[Path] = set()
    for raw in patterns:
        direct = Path(raw)
        if direct.is_file():
            matches.add(direct)
        if direct.is_absolute():
            continue
        for root in search_roots:
            if root.exists():
                matches.update(path for path in root.glob(raw) if path.is_file())
    valid = sorted(
        path for path in matches if sha256_file(path) == str(expected_sha256)
    )
    if not valid:
        observed = {
            str(path): sha256_file(path)
            for path in sorted(matches)
        }
        raise FileNotFoundError(
            f"{label} did not resolve with the pinned SHA {expected_sha256}; "
            f"observed={observed}"
        )
    return valid[0]


def validate_exp407_static_contract(
    config: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    candidates = candidate_ids(contract)
    expected_candidates = [str(item) for item in config["candidate_bank"]["order"]]
    if candidates != expected_candidates:
        raise ValueError("exp407 candidate order differs from the parent contract")
    if len(candidates) != int(config["data"]["expected_candidate_count"]):
        raise ValueError("exp407 candidate count differs from the frozen contract")
    if len(contract["legal_domains"]["primitive_pair_bank"]["candidates"]) != 11:
        raise ValueError("primitive+pair legal domain must contain 11 candidates")
    if len(contract["legal_domains"]["primitive_fixed_bank"]["candidates"]) != 7:
        raise ValueError("primitive+fixed legal domain must contain 7 candidates")
    if config["experiment"]["route"] != "ml_model":
        raise ValueError("exp407 must remain on the ml_model route")
    if int(config["features"]["feature_count"]) != 88:
        raise ValueError("exp407 must retain the corrected 88-feature schema")
    validate_inverse_rmse_weight_config(config["candidate_task_weight"])

    execution = dict(config["execution"])
    cost = {
        "active_variants": int(execution["active_variants"]),
        "objectives": int(execution["objectives"]),
        "outer_folds": int(execution["outer_folds"]),
        "planned_cpu_boosters": int(execution["planned_cpu_boosters"]),
        "parent_control_retraining": bool(execution["parent_control_retraining"]),
        "gpu_boosters": int(execution["gpu_boosters"]),
        "pf_hmm_beam_regeneration": bool(execution["pf_hmm_beam_regeneration"]),
        "inference": bool(execution["inference"]),
        "submission": bool(execution["submission"]),
    }
    expected_cost = {
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
    if cost != expected_cost:
        raise ValueError(f"exp407 Stage B cost contract changed: {cost}")
    return {
        "experiment": EXPERIMENT_NAME,
        "candidate_order": candidates,
        "candidate_count": len(candidates),
        "legal_domain_counts": [11, 7],
        "feature_count": 88,
        "cost": cost,
        "run_approved": bool(execution["run_approved"]),
    }


def _pooled_rmse(frame: pd.DataFrame, metric: str) -> float:
    rows = pd.to_numeric(frame["rows"], errors="raise").to_numpy(np.float64)
    values = pd.to_numeric(frame[metric], errors="raise").to_numpy(np.float64)
    if len(frame) == 0 or np.any(rows <= 0) or not np.isfinite(values).all():
        raise ValueError(f"cannot pool RMSE metric {metric}")
    return float(np.sqrt(np.average(np.square(values), weights=rows)))


def _hidden_like_metrics(
    new_by_well: pd.DataFrame,
    parent_by_well: pd.DataFrame,
    assignment: pd.DataFrame,
) -> pd.DataFrame:
    if assignment["well_id"].astype(str).duplicated().any():
        raise ValueError("hidden-like assignment contains duplicate well IDs")
    rows: list[dict[str, Any]] = []
    for scope, role_column in HIDDEN_LIKE_SCOPES.items():
        if role_column not in assignment:
            raise ValueError(f"hidden-like assignment is missing {role_column}")
        wells = set(
            assignment.loc[assignment[role_column].eq("valid"), "well_id"].astype(str)
        )
        new_scope = new_by_well[new_by_well["well"].astype(str).isin(wells)]
        parent_scope = parent_by_well[parent_by_well["well"].astype(str).isin(wells)]
        if set(new_scope["well"].astype(str)) != wells:
            raise ValueError(f"new by-well output does not cover every {scope} well")
        if set(parent_scope["well"].astype(str)) != wells:
            raise ValueError(f"parent by-well output does not cover every {scope} well")
        new_rmse = _pooled_rmse(new_scope, "hard_primary_rmse")
        parent_rmse = _pooled_rmse(parent_scope, "hard_primary_rmse")
        rows.append(
            {
                "scope": scope,
                "role_column": role_column,
                "wells": len(wells),
                "rows": int(new_scope["rows"].sum()),
                "parent_hard_primary_rmse": parent_rmse,
                "new_hard_primary_rmse": new_rmse,
                "delta_rmse_new_minus_parent": new_rmse - parent_rmse,
            }
        )
    return pd.DataFrame(rows)


def _bucket_comparison(
    new_bucket: pd.DataFrame,
    parent_bucket: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float]]:
    key = ["outer_fold", "distance_bucket"]
    merged = parent_bucket.merge(
        new_bucket,
        on=key,
        suffixes=("__parent", "__new"),
        validate="one_to_one",
    )
    if len(merged) != len(parent_bucket) or len(merged) != len(new_bucket):
        raise ValueError("new/parent distance-bucket inventories differ")
    for metric in ("hard_primary_rmse", "fixed_fallback_rmse"):
        merged[f"delta__{metric}__new_minus_parent"] = (
            merged[f"{metric}__new"] - merged[f"{metric}__parent"]
        )
    pooled: dict[str, float] = {}
    for bucket in ("near_0_250", "1000_plus"):
        new_scope = new_bucket[new_bucket["distance_bucket"].eq(bucket)]
        parent_scope = parent_bucket[parent_bucket["distance_bucket"].eq(bucket)]
        new_rmse = _pooled_rmse(new_scope, "hard_primary_rmse")
        parent_rmse = _pooled_rmse(parent_scope, "hard_primary_rmse")
        pooled[f"{bucket}__parent_hard_primary_rmse"] = parent_rmse
        pooled[f"{bucket}__new_hard_primary_rmse"] = new_rmse
        pooled[f"{bucket}__delta_new_minus_parent"] = new_rmse - parent_rmse
    return merged, pooled


def _verify_model_manifest(
    output_dir: Path,
    manifest: Mapping[str, Any],
    expected_count: int,
) -> dict[str, Any]:
    models = list(manifest.get("models", []))
    sha_matches = 0
    for model in models:
        path = Path(output_dir) / str(model["path"])
        if path.is_file() and sha256_file(path) == str(model["sha256"]):
            sha_matches += 1
    frame = pd.DataFrame(models)
    objective_fold_pairs = (
        set(zip(frame["objective"], frame["outer_fold"], strict=False))
        if len(frame)
        else set()
    )
    expected_pairs = {
        (objective, fold) for objective in OBJECTIVES for fold in range(5)
    }
    weight_sha_shared = False
    if len(frame) and "training_sample_weight_sha256" in frame:
        per_fold = frame.groupby("outer_fold")["training_sample_weight_sha256"].nunique()
        weight_sha_shared = bool((per_fold == 1).all())
    return {
        "model_count": len(models),
        "model_count_matches": len(models) == int(expected_count),
        "model_sha_matches": sha_matches,
        "model_sha_validation": sha_matches == int(expected_count),
        "objective_fold_inventory_matches": objective_fold_pairs == expected_pairs,
        "training_sample_weight_applied_to_all": bool(
            len(frame) and frame["training_sample_weight_applied"].astype(bool).all()
        ),
        "validation_sample_weight_applied_to_any": bool(
            len(frame) and frame["validation_sample_weight_applied"].astype(bool).any()
        ),
        "same_fold_weight_sha_shared_across_objectives": weight_sha_shared,
    }


def _validate_weight_outputs(
    output_dir: Path,
    metrics: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    manifest_path = Path(output_dir) / "candidate_task_weight_manifest.json"
    table_path = Path(output_dir) / "candidate_task_weight_by_fold.csv"
    sampling_path = Path(output_dir) / "candidate_task_weight_sampling_manifest.csv"
    truth_path = Path(output_dir) / "candidate_task_weight_truth_read_ledger.csv"
    manifest = json.loads(manifest_path.read_text())
    nested = dict(metrics["candidate_task_weight"])
    table = pd.read_csv(table_path)
    sampling = pd.read_csv(sampling_path)
    truth = pd.read_csv(truth_path)
    expected_candidates = int(config["data"]["expected_candidate_count"])
    expected_folds = int(config["validation"]["outer_folds"])
    expected_fit_rows = int(config["model"]["training"]["max_train_base_rows_per_outer_fold"])
    weight_cfg = config["candidate_task_weight"]["normalization"]
    tolerance = float(weight_cfg["final_mean_absolute_tolerance"])
    expected_partitions = set(range(expected_folds))

    sampling_contract_pass = True
    for row in sampling.itertuples(index=False):
        counts = {
            int(key): int(value)
            for key, value in json.loads(row.source_outer_fold_base_rows).items()
        }
        expected_sources = expected_partitions.difference({int(row.fit_partition)})
        if (
            set(counts) != expected_sources
            or sum(counts.values()) != expected_fit_rows
            or any(value != expected_fit_rows // (expected_folds - 1) for value in counts.values())
        ):
            sampling_contract_pass = False
            break

    forbidden_truth_columns = [
        "outer_valid_truth_reads_for_weight",
        "inner_valid_truth_reads_for_weight",
        "hidden_like_truth_reads_for_weight",
        "global_oof_truth_reads_for_weight",
        "current_test_truth_reads_for_weight",
    ]
    return {
        "manifest_sha_matches_metrics": sha256_file(manifest_path)
        == str(nested["manifest_sha256"]),
        "manifest_status_matches": manifest["status"]
        == "fit_partition_inverse_rmse_candidate_task_weights_complete",
        "partition_count_matches": int(manifest["partition_count"]) == expected_folds,
        "partition_inventory_matches": set(table["fit_partition"].astype(int))
        == expected_partitions,
        "table_rows_match": len(table) == expected_candidates * expected_folds,
        "candidate_rows_per_partition_match": bool(
            (
                table.groupby("fit_partition")["candidate_id"].nunique()
                == expected_candidates
            ).all()
        ),
        "candidate_order_stable": bool(
            all(
                group.sort_values("candidate_position")["candidate_id"].astype(str).tolist()
                == [str(item) for item in config["candidate_bank"]["order"]]
                for _, group in table.groupby("fit_partition", sort=True)
            )
        ),
        "weight_mean_matches": bool(
            (
                table.groupby("fit_partition")["final_weight"].mean().sub(1.0).abs()
                <= tolerance
            ).all()
        ),
        "weight_range_matches": bool(
            table["final_weight"].between(0.5, 1.5, inclusive="both").all()
        ),
        "weight_finite": bool(
            np.isfinite(
                table[
                    [
                        "fit_candidate_rmse",
                        "raw_inverse_rmse",
                        "mean_normalized_weight",
                        "clipped_weight",
                        "final_weight",
                    ]
                ].to_numpy(np.float64)
            ).all()
        ),
        "fit_row_counts_match": bool((table["fit_row_count"] == expected_fit_rows).all()),
        "sampling_partition_count_matches": len(sampling) == expected_folds,
        "sampling_contract_matches_parent_v5": sampling_contract_pass,
        "forbidden_truth_reads": int(
            truth[forbidden_truth_columns].to_numpy(np.int64).sum()
        ),
        "fit_valid_well_overlap": int(
            sum(int(item["fit_valid_well_overlap"]) for item in nested["audits"])
        ),
        "feature_content_sha_recorded": bool(
            all(item.get("fit_feature_content_sha256") for item in nested["audits"])
        ),
        "validation_sample_weight_applied": bool(
            manifest["validation_sample_weight_applied"]
        ),
        "metric_sample_weight_applied": bool(manifest["metric_sample_weight_applied"]),
    }


def evaluate_scientific_gate(
    *,
    new_metrics: Mapping[str, Any],
    new_fold: pd.DataFrame,
    parent_fold: pd.DataFrame,
    pooled_bucket: Mapping[str, float],
    hidden: pd.DataFrame,
    by_well: pd.DataFrame,
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    new_pooled = dict(new_metrics["pooled_score_metrics"])
    expected_error_improved_folds = int(
        (new_fold["expected_error_mae"] < parent_fold["expected_error_mae"]).sum()
    )
    logloss_nonworse_folds = int(
        (new_fold["within10_logloss"] <= parent_fold["within10_logloss"]).sum()
    )
    brier_nonworse_folds = int(
        (new_fold["within10_brier"] <= parent_fold["within10_brier"]).sum()
    )
    hard_nonworse_folds = int(
        (new_fold["hard_primary_rmse"] <= parent_fold["hard_primary_rmse"]).sum()
    )
    hidden_delta = hidden.set_index("scope")["delta_rmse_new_minus_parent"].to_dict()
    expected_hidden_scopes = set(HIDDEN_LIKE_SCOPES)
    if set(hidden_delta) != expected_hidden_scopes:
        raise ValueError("hidden-like scientific gate inventory is incomplete")
    worst_well_delta = float(by_well["delta_rmse_new_minus_parent"].max())
    checks = {
        "expected_error_mae": float(new_pooled["expected_error_mae"])
        <= float(gate["expected_error_mae_max"]),
        "expected_error_mae_improved_folds": expected_error_improved_folds
        >= int(gate["expected_error_mae_improved_fold_count_min"]),
        "within10_logloss": float(new_pooled["within10_logloss"])
        <= float(gate["within10_logloss_max"]),
        "within10_logloss_nonworse_folds": logloss_nonworse_folds
        >= int(gate["within10_logloss_nonworse_fold_count_min"]),
        "within10_brier": float(new_pooled["within10_brier"])
        <= float(gate["within10_brier_max"]),
        "within10_brier_nonworse_folds": brier_nonworse_folds
        >= int(gate["within10_brier_nonworse_fold_count_min"]),
        "hard_primary_oof_rmse": float(new_metrics["hard_primary_oof_rmse"])
        <= float(gate["hard_primary_oof_rmse_max"]),
        "hard_primary_nonworse_folds": hard_nonworse_folds
        >= int(gate["hard_primary_nonworse_fold_count_min"]),
        "near_non_regression": float(
            pooled_bucket["near_0_250__delta_new_minus_parent"]
        )
        <= float(gate["near_delta_vs_parent_max_ft"]),
        "distance_1000_plus_non_regression": float(
            pooled_bucket["1000_plus__delta_new_minus_parent"]
        )
        <= float(gate["distance_1000_plus_delta_vs_parent_max_ft"]),
        "hidden_like_spatial_non_regression": float(
            hidden_delta["hidden_like_spatial"]
        )
        <= float(gate["hidden_like_spatial_delta_vs_parent_max_ft"]),
        "hidden_like_typewell_purged_non_regression": float(
            hidden_delta["hidden_like_typewell_purged"]
        )
        <= float(gate["hidden_like_typewell_purged_delta_vs_parent_max_ft"]),
        "worst_well_non_regression": worst_well_delta
        <= float(gate["worst_well_regression_vs_parent_max_ft"]),
    }
    return {
        "metrics": {
            "new_pooled_score_metrics": new_pooled,
            "new_hard_primary_oof_rmse": float(
                new_metrics["hard_primary_oof_rmse"]
            ),
            "expected_error_improved_folds": expected_error_improved_folds,
            "within10_logloss_nonworse_folds": logloss_nonworse_folds,
            "within10_brier_nonworse_folds": brier_nonworse_folds,
            "hard_primary_nonworse_folds": hard_nonworse_folds,
            "pooled_distance_buckets": dict(pooled_bucket),
            "hidden_like_delta_rmse": hidden_delta,
            "worst_well_delta_rmse_new_minus_parent": worst_well_delta,
        },
        "checks": checks,
        "passed": bool(all(checks.values())),
    }


def evaluate_exp407_stage_b(
    *,
    config: Mapping[str, Any],
    contract: Mapping[str, Any],
    output_dir: Path,
    parent_metrics_path: Path,
    parent_bucket_path: Path,
    parent_by_well_path: Path,
    hidden_like_assignment_path: Path,
    source_config_path: Path,
    source_candidate_contract_path: Path,
) -> dict[str, Any]:
    static_contract = validate_exp407_static_contract(config, contract)
    root = Path(output_dir)
    new_metrics = json.loads((root / "selector_metrics.json").read_text())
    parent_metrics = json.loads(Path(parent_metrics_path).read_text())
    new_fold = pd.DataFrame(new_metrics["fold_metrics"]).sort_values("fold").reset_index(
        drop=True
    )
    parent_fold = pd.DataFrame(parent_metrics["fold_metrics"]).sort_values(
        "fold"
    ).reset_index(drop=True)
    if not np.array_equal(new_fold["fold"].to_numpy(), parent_fold["fold"].to_numpy()):
        raise ValueError("new and parent selector folds differ")
    if not np.array_equal(new_fold["rows"].to_numpy(), parent_fold["rows"].to_numpy()):
        raise ValueError("new and parent selector fold row counts differ")

    fold_comparison = parent_fold.add_prefix("parent__").join(new_fold.add_prefix("new__"))
    fold_comparison["fold"] = new_fold["fold"].to_numpy()
    comparison_metrics = [
        "expected_error_mae",
        "within10_logloss",
        "within10_brier",
        "hard_primary_rmse",
    ]
    for metric in comparison_metrics:
        fold_comparison[f"delta__{metric}__new_minus_parent"] = (
            new_fold[metric] - parent_fold[metric]
        )
    fold_path = root / "exp407_parent_fold_comparison.csv"
    fold_comparison.to_csv(fold_path, index=False)

    new_bucket = pd.read_csv(root / "selector_distance_bucket_metrics.csv")
    parent_bucket = pd.read_csv(parent_bucket_path)
    bucket_comparison, pooled_bucket = _bucket_comparison(new_bucket, parent_bucket)
    bucket_path = root / "exp407_parent_bucket_comparison.csv"
    bucket_comparison.to_csv(bucket_path, index=False)

    new_by_well = pd.read_csv(root / "selector_by_well.csv", dtype={"well": str})
    parent_by_well = pd.read_csv(parent_by_well_path, dtype={"well": str})
    by_well = parent_by_well.merge(
        new_by_well,
        on="well",
        suffixes=("__parent", "__new"),
        validate="one_to_one",
    )
    if len(by_well) != int(config["data"]["expected_wells"]):
        raise ValueError("new/parent by-well comparison is incomplete")
    if not np.array_equal(by_well["rows__parent"], by_well["rows__new"]):
        raise ValueError("new/parent by-well row counts differ")
    by_well["delta_rmse_new_minus_parent"] = (
        by_well["hard_primary_rmse__new"]
        - by_well["hard_primary_rmse__parent"]
    )
    by_well_path = root / "exp407_parent_by_well_comparison.csv"
    by_well.to_csv(by_well_path, index=False)

    assignment = pd.read_csv(hidden_like_assignment_path, dtype={"well_id": str})
    hidden = _hidden_like_metrics(new_by_well, parent_by_well, assignment)
    hidden_path = root / "exp407_parent_hidden_like_comparison.csv"
    hidden.to_csv(hidden_path, index=False)

    feature_schema = json.loads((root / "feature_schema.json").read_text())
    candidate_metrics = pd.read_csv(root / "selector_candidate_metrics.csv")
    selection = pd.read_csv(root / "selector_selection_rate.csv")
    model_manifest = json.loads((root / "selector_model_manifest.json").read_text())
    model_audit = _verify_model_manifest(
        root,
        model_manifest,
        int(config["gates"]["technical"]["expected_model_count"]),
    )
    weight_audit = _validate_weight_outputs(root, new_metrics, config)
    positive_weight_checks = [
        "manifest_sha_matches_metrics",
        "manifest_status_matches",
        "partition_count_matches",
        "partition_inventory_matches",
        "table_rows_match",
        "candidate_rows_per_partition_match",
        "candidate_order_stable",
        "weight_mean_matches",
        "weight_range_matches",
        "weight_finite",
        "fit_row_counts_match",
        "sampling_partition_count_matches",
        "sampling_contract_matches_parent_v5",
        "feature_content_sha_recorded",
    ]

    expected_candidates = int(config["data"]["expected_candidate_count"])
    expected_base_rows = int(config["data"]["expected_base_rows"])
    candidate_rows = candidate_metrics.groupby("candidate_id")["rows"].sum()
    selected_rows = selection.groupby("objective")["selected_rows"].sum()
    technical_checks = {
        "base_rows": int(new_metrics["compact_meta_oof_rows"]) == expected_base_rows,
        "candidate_count": candidate_metrics["candidate_id"].nunique()
        == expected_candidates,
        "candidate_long_rows": int(new_metrics["candidate_score_oof_rows"])
        == int(config["data"]["expected_candidate_long_oof_rows"]),
        "candidate_rows_each": bool(
            len(candidate_rows) == expected_candidates
            and (candidate_rows == int(config["data"]["expected_candidate_rows_each"])).all()
        ),
        "selection_inventory": bool(
            len(selected_rows) == 2 and (selected_rows == expected_base_rows).all()
        ),
        "feature_count": int(feature_schema["feature_count"])
        == int(config["data"]["expected_feature_count"]),
        "feature_schema_logical_sha256": str(feature_schema["feature_schema_sha256"])
        == str(config["data"]["expected_feature_schema_logical_sha256"]),
        "candidate_order": model_manifest["candidate_order"]
        == [str(item) for item in config["candidate_bank"]["order"]],
        "legal_domain_counts": static_contract["legal_domain_counts"]
        == list(config["gates"]["technical"]["legal_domain_counts"]),
        "weight_manifest": bool(
            all(bool(weight_audit[key]) for key in positive_weight_checks)
        )
        and int(weight_audit["forbidden_truth_reads"]) == 0
        and int(weight_audit["fit_valid_well_overlap"]) == 0
        and not bool(weight_audit["validation_sample_weight_applied"])
        and not bool(weight_audit["metric_sample_weight_applied"]),
        "models": bool(
            model_audit["model_count_matches"]
            and model_audit["model_sha_validation"]
            and model_audit["objective_fold_inventory_matches"]
            and model_audit["training_sample_weight_applied_to_all"]
            and not model_audit["validation_sample_weight_applied_to_any"]
            and model_audit["same_fold_weight_sha_shared_across_objectives"]
        ),
        "candidate_score_sha": sha256_file(root / "candidate_score_oof.parquet")
        == str(new_metrics["candidate_score_oof_sha256"]),
        "model_manifest_sha": sha256_file(root / "selector_model_manifest.json")
        == str(new_metrics["model_manifest_sha256"]),
        "source_config_sha_recorded": bool(sha256_file(source_config_path)),
        "candidate_contract_sha_recorded": bool(
            sha256_file(source_candidate_contract_path)
        ),
        "parent_control_retraining": not bool(
            config["execution"]["parent_control_retraining"]
        ),
    }

    parent_control = config["parent_control"]
    for key in (
        "expected_error_mae",
        "within10_logloss",
        "within10_brier",
        "hard_primary_oof_rmse",
    ):
        actual_key = (
            "hard_primary_oof_rmse"
            if key == "hard_primary_oof_rmse"
            else key
        )
        actual = (
            parent_metrics[actual_key]
            if actual_key in parent_metrics
            else parent_metrics["pooled_score_metrics"][actual_key]
        )
        if not math.isclose(
            float(actual),
            float(parent_control[key]),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError(f"pinned parent metric changed: {key}")

    scientific = evaluate_scientific_gate(
        new_metrics=new_metrics,
        new_fold=new_fold,
        parent_fold=parent_fold,
        pooled_bucket=pooled_bucket,
        hidden=hidden,
        by_well=by_well,
        gate=config["gates"]["scientific"],
    )
    technical_passed = bool(all(technical_checks.values()))
    scientific_passed = bool(scientific["passed"])
    if not technical_passed:
        decision = "technical_error_same_frozen_contract_only"
    elif scientific_passed:
        decision = "pass_stage_c_eligible_pending_separate_approval"
    else:
        decision = "fail_close_exp407_without_rescue"

    summary = {
        "status": "exp407_stage_b_gate_evaluated",
        "experiment": EXPERIMENT_NAME,
        "route": config["experiment"]["route"],
        "source_config_sha256": sha256_file(source_config_path),
        "source_candidate_contract_sha256": sha256_file(
            source_candidate_contract_path
        ),
        "parent_inputs": {
            "selector_metrics": {
                "path": str(parent_metrics_path),
                "sha256": sha256_file(parent_metrics_path),
            },
            "distance_bucket_metrics": {
                "path": str(parent_bucket_path),
                "sha256": sha256_file(parent_bucket_path),
            },
            "by_well_metrics": {
                "path": str(parent_by_well_path),
                "sha256": sha256_file(parent_by_well_path),
            },
            "hidden_like_assignment": {
                "path": str(hidden_like_assignment_path),
                "sha256": sha256_file(hidden_like_assignment_path),
            },
        },
        "cost_contract": static_contract["cost"],
        "technical": {
            "checks": technical_checks,
            "weight_audit": weight_audit,
            "model_audit": model_audit,
            "passed": technical_passed,
        },
        "scientific": scientific,
        "decision": decision,
        "artifacts": {
            "fold_comparison": fold_path.name,
            "bucket_comparison": bucket_path.name,
            "hidden_like_comparison": hidden_path.name,
            "by_well_comparison": by_well_path.name,
            "candidate_metrics": "selector_candidate_metrics.csv",
            "selection_rate": "selector_selection_rate.csv",
            "calibration": "selector_calibration.csv",
            "candidate_task_weight_manifest": "candidate_task_weight_manifest.json",
        },
    }
    gate_path = root / "exp407_stage_b_gate.json"
    write_json(gate_path, summary)
    gate_sha256 = sha256_file(gate_path)
    reproducibility_path = root / "reproducibility_manifest.json"
    reproducibility = json.loads(reproducibility_path.read_text())
    reproducibility.update(
        {
            "status": "exp407_stage_b_gate_evaluated",
            "source_config_sha256": summary["source_config_sha256"],
            "source_candidate_contract_sha256": summary[
                "source_candidate_contract_sha256"
            ],
            "exp407_stage_b_gate_sha256": gate_sha256,
            "exp407_decision": decision,
        }
    )
    write_json(reproducibility_path, reproducibility)
    summary["gate_file_sha256"] = gate_sha256
    return summary


__all__ = [
    "EXPERIMENT_NAME",
    "evaluate_exp407_stage_b",
    "evaluate_scientific_gate",
    "resolve_pinned_input",
    "validate_exp407_static_contract",
]
