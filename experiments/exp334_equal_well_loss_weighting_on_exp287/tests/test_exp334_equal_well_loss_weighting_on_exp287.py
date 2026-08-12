from __future__ import annotations

import copy
import hashlib
import importlib.util
import inspect
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = ROOT / "experiments" / "exp334_equal_well_loss_weighting_on_exp287"
TRAIN_PATH = EXP_DIR / (
    "exp334_equal_well_loss_weighting_on_exp287_compact_selfcontained_train.py"
)
INFERENCE_PATH = EXP_DIR / (
    "exp334_equal_well_loss_weighting_on_exp287_compact_selfcontained_inference.py"
)


def load_module(path: Path, name: str):
    os.environ["EXP334_IMPORT_ONLY"] = "1"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


train = load_module(TRAIN_PATH, "exp334_train")
inference = load_module(INFERENCE_PATH, "exp334_inference")
CONFIG = yaml.safe_load((EXP_DIR / "config.yaml").read_text())


def test_train_completed_guard_failed_and_all_execution_is_closed() -> None:
    contract = train.validate_scientific_contract(CONFIG, require_train_approval=False)
    assert CONFIG["experiment"]["route"] == "ml_model"
    assert CONFIG["experiment"]["status"] == (
        "train_complete_guard_failed_closed_no_inference"
    )
    assert CONFIG["execution"]["stage"] == "equal_well_weight_train"
    assert CONFIG["execution"]["implementation_approved"] is True
    assert CONFIG["execution"]["canonical_train_notebook_adoption_approved"] is True
    assert CONFIG["execution"]["zero_booster_preflight_approved"] is True
    assert CONFIG["execution"]["fifteen_booster_train_approved"] is True
    assert CONFIG["execution"]["preflight_completed"] is True
    assert CONFIG["execution"]["preflight_kernel_version"] == 1
    assert CONFIG["execution"]["preflight_kernel_id_no"] == 128110184
    assert CONFIG["execution"]["train_kernel_version"] == 2
    assert CONFIG["execution"]["train_kernel_id_no"] == 128110184
    assert CONFIG["execution"]["train_completed"] is True
    assert CONFIG["execution"]["promotion_guard_passed"] is False
    assert CONFIG["execution"]["kaggle_push_approved"] is False
    assert CONFIG["execution"]["run_train"] is False
    assert CONFIG["execution"]["run_inference"] is False
    assert CONFIG["execution"]["create_submission"] is False
    assert CONFIG["execution"]["submit_to_kaggle"] is False
    assert contract["planned_gpu_boosters"] == 15
    assert contract["control_retraining_boosters"] == 0
    assert contract["stage"] == "equal_well_weight_train"
    with pytest.raises(RuntimeError, match="separate push approval"):
        train.validate_scientific_contract(CONFIG, require_train_approval=True)


def test_gpu_cost_is_one_variant_three_configs_five_folds() -> None:
    assert CONFIG["model"]["execution_count"] == {
        "active_variants": 1,
        "lightgbm_configs": 3,
        "folds": 5,
        "planned_gpu_boosters": 15,
        "control_retraining_boosters": 0,
    }
    assert CONFIG["model"]["source_surface"]["lightgbm_config_indices"] == [0, 1, 2]
    assert CONFIG["model"]["source_surface"]["final_feature_count"] == 421
    assert CONFIG["model"]["train_weight"]["validation_weight"] is None


def test_equal_well_weight_formula_and_invariants() -> None:
    row_ids = ["a0", "a1", "a2", "b0", "c0", "c1"]
    wells = ["a", "a", "a", "b", "c", "c"]
    weights, summary, evidence = train.build_equal_well_weights(row_ids, wells)
    np.testing.assert_allclose(weights, [2 / 3, 2 / 3, 2 / 3, 2.0, 1.0, 1.0])
    assert weights.mean() == pytest.approx(1.0, abs=1.0e-12)
    totals = summary.set_index("well")["total_weight"]
    np.testing.assert_allclose(totals.to_numpy(), np.full(3, 2.0), atol=1.0e-12)
    assert evidence["expected_total_weight_per_well"] == 2.0
    assert evidence["validation_weight"] is None
    assert evidence["target_or_error_input_used"] is False
    assert len(evidence["row_identity_sha256"]) == 64
    assert len(evidence["row_weight_logical_sha256"]) == 64


def test_weight_builder_has_no_target_error_or_validation_input() -> None:
    parameters = set(inspect.signature(train.build_equal_well_weights).parameters)
    assert parameters == {"row_ids", "wells", "tolerance"}
    assert not parameters.intersection(
        {"target", "prediction", "error", "formation", "outer_valid", "valid_weight"}
    )
    with pytest.raises(ValueError, match="unique"):
        train.build_equal_well_weights(["dup", "dup"], ["a", "a"])


def test_train_source_changes_only_train_sample_weight_contract() -> None:
    source = TRAIN_PATH.read_text()
    assert "sample_weight=train_weights" in source
    assert "eval_set=[(x_valid, target[valid_indices])]" in source
    assert "eval_metric=\"rmse\"" in source
    assert "eval_sample_weight" not in source
    assert "validation_weight\": None" in source
    assert "build_fold_formation_surface" not in source
    assert "run_fold_safe_formation_train" not in source
    assert "fold_safe_formation_oof_predictions.parquet" in source
    assert "verify_parent_artifacts" in source


def test_promotion_guard_uses_unweighted_exp287_and_clean_tail_controls(
    tmp_path: Path,
) -> None:
    wells = np.repeat([f"w{fold}" for fold in range(5)], 2)
    folds = np.repeat(np.arange(5, dtype=np.int8), 2)
    rows = len(wells)
    truth = 100.0 + np.arange(rows, dtype=np.float32)
    base = pd.DataFrame(
        {
            "id": [f"row_{index}" for index in range(rows)],
            "well": wells,
            "last_known_tvt": np.full(rows, 100.0, dtype=np.float32),
            "target": truth - 100.0,
            "md_since": np.asarray([100.0, 200.0, 500.0, 700.0, 1200.0] * 2),
        }
    )
    exp287 = pd.DataFrame(
        {
            "id": base["id"],
            "well": wells,
            "outer_fold": folds,
            "fold_safe_formation_74_addonly__lgb_mean__pred_tvt": truth,
        }
    )
    exp264 = pd.DataFrame(
        {
            "id": base["id"],
            "well": wells,
            "selector_compact_addonly__lgb_mean__pred_tvt": truth,
            "matched_control__lgb_mean__pred_tvt": truth,
        }
    )
    assignment_path = tmp_path / "hidden.csv"
    pd.DataFrame(
        {
            "well_id": [f"w{fold}" for fold in range(5)],
            "verification_like_spatial_role": ["valid"] * 5,
            "verification_like_typewell_purged_role": ["valid"] * 5,
        }
    ).to_csv(assignment_path, index=False)
    config = copy.deepcopy(CONFIG)
    config["data"]["hidden_like_assignment_sha256"] = hashlib.sha256(
        assignment_path.read_bytes()
    ).hexdigest()
    guard, fold_metrics, scope_metrics, hidden_metrics, by_well = (
        train.evaluate_promotion_guards(
            config=config,
            base_frame=base,
            exp287_control=exp287,
            exp264_control=exp264,
            oof_fold=folds,
            new_prediction=truth,
            hidden_like_assignment_path=assignment_path,
        )
    )
    assert guard["passed"] is True
    assert guard["metric_weighting"] == "unweighted_rows"
    assert guard["nonworse_folds_vs_exp287"] == 5
    assert guard["by_well_delta_p95_vs_exp287"] == 0.0
    assert guard["worst_well_delta_rmse_vs_exp264"] == 0.0
    assert len(fold_metrics) == 5
    assert set(scope_metrics["scope"]) == {
        "all",
        "near_0_250",
        "mid_250_1000",
        "1000_plus",
    }
    assert len(hidden_metrics) == 2
    assert len(by_well) == 5


def test_parent_sha_and_feature_contract_match_recorded_exp287() -> None:
    parent_metrics = json.loads(
        (
            ROOT
            / "experiments"
            / "exp287_fold_safe_formation_74_addonly_on_exp264"
            / "metrics.json"
        ).read_text()
    )
    data = CONFIG["data"]
    assert data["expected_exp287_oof_sha256"] == parent_metrics["artifact_sha256"]["oof"]
    assert data["expected_exp287_model_manifest_sha256"] == (
        parent_metrics["artifact_sha256"]["model_manifest"]
    )
    assert data["expected_exp287_metrics_sha256"] == parent_metrics["artifact_sha256"][
        "metrics"
    ]
    assert data["expected_exp287_fold_metrics_sha256"] == (
        parent_metrics["artifact_sha256"]["fold_metrics"]
    )
    assert data["expected_exp287_by_well_sha256"] == parent_metrics["artifact_sha256"][
        "by_well"
    ]
    assert data["expected_exp287_formation_fold_manifest_sha256"] == (
        parent_metrics["formation_fold_manifest_sha256"]
    )
    assert data["expected_exp287_raw_schema_audit_sha256"] == (
        parent_metrics["raw_schema_audit_sha256"]
    )
    parent_config = yaml.safe_load(
        (
            ROOT
            / "experiments"
            / "exp287_fold_safe_formation_74_addonly_on_exp264"
            / "config.yaml"
        ).read_text()
    )
    for key in [
        "stage_c_expected_nested_selector_metrics_sha256",
        "stage_c_expected_nested_selector_model_manifest_sha256",
        "stage_c_expected_nested_compact_manifest_sha256",
        "stage_c_expected_compact_meta_schema_file_sha256",
        "stage_c_expected_compact_meta_schema_logical_sha256",
    ]:
        assert data[key] == parent_config["data"][key]


def test_train_kernel_sources_include_frozen_parent_output() -> None:
    sources = CONFIG["runtime"]["kaggle"]["train_kernel_sources"]
    assert sources == [
        "kentookumura/exp264-exp263-confidence-dual-selector-train",
        "kentookumura/exp264-exp263-confidence-dual-selector-tvt-train",
        "kentookumura/exp072-exp063-full-replay-feature-cache-train",
        "kentookumura/exp145-train",
        "kentookumura/exp287-foldsafe-form74-addonly-exp264-train",
    ]


def test_inference_candidate_is_fail_closed() -> None:
    status = inference.validate_inference_is_closed(CONFIG)
    assert status["prediction_generated"] is False
    assert status["submission_generated"] is False
    changed = copy.deepcopy(CONFIG)
    changed["execution"]["run_inference"] = True
    with pytest.raises(RuntimeError, match="not implemented or authorized"):
        inference.validate_inference_is_closed(changed)
    source = INFERENCE_PATH.read_text()
    assert "submission.csv" not in source
    assert "copyfile" not in source
    assert "raise RuntimeError" in source


def test_compact_sources_are_notebook_safe_and_candidates_exist() -> None:
    train_source = TRAIN_PATH.read_text()
    inference_source = INFERENCE_PATH.read_text()
    assert "__file__" not in train_source
    assert "__file__" not in inference_source
    assert "# ## Contents" in train_source
    assert "# ## 7. Weighted LightGBM training" in train_source
    assert "# ## 8. Unweighted OOF and promotion guards" in train_source
    assert "# ## Contents" in inference_source
    assert (EXP_DIR / TRAIN_PATH.with_suffix(".ipynb").name).exists()
    assert (EXP_DIR / INFERENCE_PATH.with_suffix(".ipynb").name).exists()
