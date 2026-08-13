from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src import feature_union_pipeline
from src.candidate_selector_pipeline import KEY_COLUMNS
from src.feature_union_pipeline import (
    assemble_union_matrix,
    audit_added_feature_relationships,
    evaluate_union_guards,
    freeze_union_feature_schema,
    union_cost_contract,
    validate_role_alignment,
)
from src.fold_safe_formation_pipeline import canonical_formation_feature_names
from tests.test_support import require_saved_files

EXP372 = Path("experiments/exp372_exp287_exp335_feature_union_on_exp264")


def load_config() -> dict:
    return yaml.safe_load((EXP372 / "config.yaml").read_text())


def feature_groups() -> tuple[list[str], list[str], list[str], list[str]]:
    clean = ["last_known_tvt", *[f"clean_{index}" for index in range(272)]]
    parent = [f"parent_{index}" for index in range(74)]
    formation = canonical_formation_feature_names()
    signed = [f"signed_{index}" for index in range(23)]
    return clean, parent, formation, signed


def frozen_contract() -> tuple[list[str], dict]:
    clean, parent, formation, signed = feature_groups()
    return freeze_union_feature_schema(
        clean_features=clean,
        parent_features=parent,
        formation_features=formation,
        signed_features=signed,
        forbidden_columns=["target", "error", "true_tvt"],
    )


def test_cost_contract_is_one_union_variant_and_fifteen_new_boosters() -> None:
    config = load_config()
    contract = union_cost_contract(config)
    assert contract["active_variants"] == [
        "formation74_signed23_union_addonly"
    ]
    assert contract["lightgbm_config_indices"] == [0, 1, 2]
    assert contract["folds"] == 5
    assert contract["planned_gpu_boosters"] == 15
    assert contract["parent_control_retraining_boosters"] == 0
    assert contract["standalone_parent_retraining_boosters"] == 0
    assert contract["selector_retraining_boosters"] == 0
    assert contract["feature_generation_runs"] == 0
    assert contract["feature_counts"]["final"] == 444


def test_feature_schema_freezes_exact_group_order_before_truth() -> None:
    clean, parent, formation, signed = feature_groups()
    features, contract = frozen_contract()
    assert features == [*clean, *parent, *formation, *signed]
    assert len(features) == len(set(features)) == 444
    assert contract["frozen_order"] == [
        "clean_base",
        "saved_exp264_compact",
        "fold_safe_formation",
        "signed_residual_compact",
    ]
    assert contract["truth_or_error_loaded_before_schema_freeze"] == 0

    with pytest.raises(ValueError, match="forbidden"):
        freeze_union_feature_schema(
            clean_features=[*clean[:-1], "target"],
            parent_features=parent,
            formation_features=formation,
            signed_features=signed,
            forbidden_columns=["target"],
        )


def test_parent_loader_adapts_verified_feature_key_to_exp264_loader_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_train = pd.DataFrame({"id": ["train"]})
    expected_valid = pd.DataFrame({"id": ["valid"]})
    observed: dict[str, object] = {}

    def fake_load_stage_d_compact_fold(
        *,
        stage_c_root: Path,
        stage_c_evidence: dict,
        downstream_outer_fold: int,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        observed["root"] = stage_c_root
        observed["evidence"] = stage_c_evidence
        observed["fold"] = downstream_outer_fold
        return expected_train, expected_valid

    monkeypatch.setattr(
        feature_union_pipeline,
        "load_stage_d_compact_fold",
        fake_load_stage_d_compact_fold,
    )
    features = [f"compact_{index}" for index in range(74)]
    train, valid = feature_union_pipeline.load_parent_compact_fold(
        {"root": "/tmp/exp264-stage-c", "features": features, "partitions": []},
        downstream_outer_fold=3,
    )
    assert train is expected_train
    assert valid is expected_valid
    assert observed["root"] == Path("/tmp/exp264-stage-c")
    assert observed["fold"] == 3
    assert observed["evidence"]["compact_features"] == features


def synthetic_role_frames(rows: int = 6) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    clean, parent_features, formation_features, signed_features = feature_groups()
    ids = [f"row_{index}" for index in range(rows)]
    wells = [f"well_{index // 2}" for index in range(rows)]
    base = pd.DataFrame(
        {
            "id": ids,
            "well": wells,
            **{
                feature: np.arange(rows, dtype=np.float32) + position
                for position, feature in enumerate(clean)
            },
        }
    )
    metadata = {
        "id": ids,
        "well": wells,
        "well_row_idx": np.arange(rows),
        "outer_fold": np.zeros(rows, dtype=np.int8),
        "md_since": np.arange(rows, dtype=np.float32),
        "last_known_tvt": np.full(rows, 1000.0, dtype=np.float32),
    }
    parent = pd.DataFrame(
        {
            **metadata,
            **{
                feature: np.arange(rows, dtype=np.float32) + 1000 + position
                for position, feature in enumerate(parent_features)
            },
        }
    )
    formation = pd.DataFrame(
        {
            "id": ids,
            "well": wells,
            **{
                feature: np.arange(rows, dtype=np.float32) + 2000 + position
                for position, feature in enumerate(formation_features)
            },
        }
    )
    signed = pd.DataFrame(
        {
            **metadata,
            **{
                feature: np.arange(rows, dtype=np.float32) + 3000 + position
                for position, feature in enumerate(signed_features)
            },
        }
    )
    return base, parent, formation, signed


def test_role_alignment_and_matrix_assembly_preserve_444_order() -> None:
    clean, parent_features, formation_features, signed_features = feature_groups()
    base, parent, formation, signed = synthetic_role_frames()
    validate_role_alignment(
        role="train",
        parent=parent,
        formation=formation,
        signed=signed,
    )
    indices, values = assemble_union_matrix(
        base_frame=base,
        base_index=pd.Index(base["id"]),
        base_features=clean,
        parent=parent,
        parent_features=parent_features,
        formation=formation,
        formation_features=formation_features,
        signed=signed,
        signed_features=signed_features,
        chunk_columns=32,
    )
    np.testing.assert_array_equal(indices, np.arange(len(base)))
    assert values.shape == (len(base), 444)
    np.testing.assert_array_equal(values[:, :273], base[clean].to_numpy(np.float32))
    np.testing.assert_array_equal(
        values[:, 273:347], parent[parent_features].to_numpy(np.float32)
    )
    np.testing.assert_array_equal(
        values[:, 347:421], formation[formation_features].to_numpy(np.float32)
    )
    np.testing.assert_array_equal(
        values[:, 421:], signed[signed_features].to_numpy(np.float32)
    )

    drifted = signed.iloc[::-1].reset_index(drop=True)
    with pytest.raises(ValueError, match="key alignment"):
        validate_role_alignment(
            role="train",
            parent=parent,
            formation=formation,
            signed=drifted,
        )


def test_relationship_audit_reports_duplicate_without_pruning() -> None:
    features, contract = frozen_contract()
    rng = np.random.default_rng(42)
    train = rng.normal(size=(24, len(features))).astype(np.float32)
    valid = rng.normal(size=(12, len(features))).astype(np.float32)
    formation_feature = contract["feature_groups"]["fold_safe_formation"][0]
    clean_feature = contract["feature_groups"]["clean_base"][1]
    formation_index = features.index(formation_feature)
    clean_index = features.index(clean_feature)
    train[:, formation_index] = train[:, clean_index]
    valid[:, formation_index] = valid[:, clean_index]
    audit = audit_added_feature_relationships(
        train_values=train,
        valid_values=valid,
        feature_contract=contract,
        sample_rows=36,
    )
    row = audit.set_index("feature").loc[formation_feature]
    assert row["exact_duplicate_count"] == 1
    assert clean_feature in row["exact_duplicate_features"]
    assert row["policy"] == "report_only_no_pruning"


def test_incremental_and_tail_gates_are_independent_and_both_required(
    tmp_path: Path,
) -> None:
    config = copy.deepcopy(load_config())
    rows = 30
    wells = [f"well_{index // 3}" for index in range(rows)]
    folds = np.asarray([index % 5 for index in range(rows)], dtype=np.int8)
    truth = np.linspace(1000.0, 1100.0, rows, dtype=np.float32)
    base = pd.DataFrame(
        {
            "id": [f"row_{index}" for index in range(rows)],
            "well": wells,
            "md_since": np.resize(
                np.asarray([100.0, 500.0, 1500.0], dtype=np.float32), rows
            ),
        }
    )
    controls = pd.DataFrame(
        {
            "id": base["id"],
            "well": wells,
            "outer_fold": folds,
            "actual_tvt": truth,
            "clean273": truth + 2.0,
            "exp264": truth + 2.0,
            "exp287": truth + np.where(folds % 2 == 0, 1.0, 1.2),
            "exp335": truth + np.where(folds % 2 == 0, 1.2, 1.0),
        }
    )
    assignment = pd.DataFrame(
        {
            "well_id": sorted(set(wells)),
            "verification_like_spatial_role": "valid",
            "verification_like_typewell_purged_role": "valid",
        }
    )
    assignment_path = tmp_path / "hidden.csv"
    assignment.to_csv(assignment_path, index=False)
    config["data"]["hidden_like_assignment_sha256"] = hashlib.sha256(
        assignment_path.read_bytes()
    ).hexdigest()
    importance = pd.DataFrame(
        [
            {
                "outer_fold": fold,
                "model": "lgb0",
                "importance_type": "gain",
                "feature": f"{group}_{fold}",
                "feature_group": group,
                "importance": 1.0,
            }
            for fold in range(5)
            for group in ("fold_safe_formation", "signed_residual_compact")
        ]
    )
    union = truth + np.float32(0.1)
    incremental, tail, *_ = evaluate_union_guards(
        config=config,
        base_frame=base,
        controls=controls,
        oof_fold=folds,
        new_prediction=union,
        hidden_like_assignment_path=assignment_path,
        importance=importance,
    )
    assert incremental["passed"] is True
    assert tail["passed"] is True

    no_signed_gain = importance[
        importance["feature_group"].ne("signed_residual_compact")
    ].copy()
    incremental, tail, *_ = evaluate_union_guards(
        config=config,
        base_frame=base,
        controls=controls,
        oof_fold=folds,
        new_prediction=union,
        hidden_like_assignment_path=assignment_path,
        importance=no_signed_gain,
    )
    assert incremental["checks"]["signed_total_gain_positive"] is False
    assert incremental["passed"] is False
    assert tail["passed"] is True


def test_train_completed_failed_closed_before_separate_inference_override() -> None:
    config = load_config()
    assert config["inference"]["train_result_status"] == (
        "train_complete_guard_failed_closed"
    )
    assert config["implementation"]["kaggle_package_created"] is True
    assert config["implementation"]["enabled"] is True
    assert config["implementation"]["jupytext_source_created"] is True
    assert config["implementation"]["tests_created"] is True
    assert config["execution"]["implementation_approved"] is True
    assert config["execution"]["canonical_train_notebook_adoption_approved"] is True
    assert config["execution"]["canonical_train_notebook_sha256"] == (
        "33f0978521812059e65c80b80da21bb2b7f661da28c05de6306275c5201cc95a"
    )
    assert config["execution"]["kaggle_package_approved"] is True
    assert config["execution"]["kaggle_push_approved"] is False
    assert config["execution"]["train_run_approved"] is False
    assert config["execution"]["run_train"] is False
    assert config["execution"]["train_kernel_version"] == 2
    assert config["execution"]["planned_train_kernel_version"] == 2
    assert config["execution"]["technical_retry_approved"] is True
    assert config["execution"]["technical_retry_package_prepared"] is True
    assert config["execution"]["last_train_run"]["kernel_id_no"] == 128530478
    assert config["execution"]["current_train_run"]["status"] == "complete"
    assert config["execution"]["current_train_run"]["kernel_id_no"] == 128530478
    assert config["execution"]["current_train_run"]["gpu_boosters_completed"] == 15
    assert config["execution"]["current_train_run"]["pooled_cv_rmse"] == pytest.approx(
        8.071563864946972
    )
    assert config["execution"]["current_train_run"]["technical_gate_passed"] is True
    assert (
        config["execution"]["current_train_run"][
            "incremental_utility_gate_passed"
        ]
        is False
    )
    assert (
        config["execution"]["current_train_run"]["tail_promotion_gate_passed"]
        is False
    )
    assert config["execution"]["current_train_run"]["promotion_gate_passed"] is False
    assert config["execution"]["last_train_run"]["gpu_boosters_started"] == 0
    assert config["execution"]["submission_approved"] is False
    assert config["execution"]["submit_to_kaggle"] is False

    source = (
        EXP372
        / "exp372_exp287_exp335_feature_union_on_exp264_compact_selfcontained_train.py"
    ).read_text()
    assert "run_feature_union_train(" in source
    assert "planned_gpu_boosters" in source
    assert "saved_control_retraining" in source
    assert "Inference executed: False" in source
    assert "Submission generated or submitted: False" in source
    assert "__file__" not in source
    assert source.count("# %% [markdown]") >= 9


def test_guard_failed_inference_override_is_cpu_zero_fit_and_submit_disarmed() -> None:
    config = load_config()
    inference = config["inference"]
    execution = config["execution"]

    assert config["experiment"]["status"] == execution["stage"]
    assert execution["inference_approved"] is True
    assert execution["inference_guard_override"] is True
    assert execution["canonical_inference_notebook_adoption_approved"] is True
    canonical_inference = (
        EXP372 / "exp372_exp287_exp335_feature_union_on_exp264_inference.ipynb"
    )
    assert hashlib.sha256(canonical_inference.read_bytes()).hexdigest() == (
        execution["canonical_inference_notebook_sha256"]
    )
    assert execution["inference_package_approved"] is True
    assert execution["inference_push_approved"] is False
    assert execution["inference_run_approved"] is False
    assert execution["planned_inference_kernel_version"] == (
        execution["current_inference_run"]["kernel_version"]
    )
    assert execution["inference_package_cells"] == 19
    assert execution["inference_package_support_files"] == 42
    assert execution["inference_kernel_source_count"] == 9
    assert execution["inference_dataset_source_count"] == 1
    assert execution["run_inference"] is False
    assert execution["create_submission"] is False
    assert execution["submission_approved"] is False
    assert execution["submit_to_kaggle"] is False
    assert inference["explicit_user_override_after_guard_failure"] is True
    assert inference["technical_gate_passed"] is True
    assert inference["incremental_utility_gate_passed"] is False
    assert inference["tail_promotion_gate_passed"] is False
    assert inference["promotion_gate_passed"] is False
    assert inference["runtime"] == "kaggle_cpu"
    assert inference["booster_training_count"] == 0
    assert inference["parent_selector_model_count"] == 40
    assert inference["signed_selector_model_count"] == 20
    assert inference["tvt_model_count"] == 15
    assert inference["selector_predict_base_row_chunk_size"] == 20000
    assert inference["signed_top1_value_parity_atol"] == pytest.approx(1.0e-5)
    assert inference["tvt_model_manifest_status"] == "completed_15_gpu_boosters"
    assert inference["expected_final_feature_count"] == 444
    assert inference["generate_submission_file"] is True
    assert inference["competition_submit_authorized"] is False
    assert inference["submit_to_kaggle"] is False
    assert config["features"]["schema_source"] == (
        "corrected_exp264_stage_a_v4_fixed_88"
    )
    assert config["features"]["shape_windows"] == [32, 128, 512]
    assert config["features"]["raw_context"]["horizontal_numeric_allowlist"] == [
        "MD",
        "X",
        "Y",
        "Z",
        "GR",
    ]
    assert config["features"]["raw_context"]["forbidden_columns"] == [
        "TVT",
        "TVT_input",
        "target",
        "true_tvt",
    ]

    manifest_path = (
        EXP372 / "kaggle/output/train_v2/artifacts/model_manifest.json"
    )
    require_saved_files(manifest_path)
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == (
        inference["tvt_model_manifest_sha256"]
    )
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == inference["tvt_model_manifest_status"]


def test_inference_source_preserves_union_order_and_regenerates_raw_test() -> None:
    source_path = (
        EXP372
        / "exp372_exp287_exp335_feature_union_on_exp264_compact_selfcontained_inference.py"
    )
    source = source_path.read_text()

    assert "build_current_test_formation_surface(" in source
    assert "current_test_bundle_from_wide(" in source
    assert "build_signed_compact_meta(" in source
    assert "final_feature_columns" in source
    assert "booster_training_count" in source
    assert "competition_submit_authorized" in source
    assert "submission.csv is generated for submit-check" in source
    assert 'inference_cfg["tvt_model_manifest_status"]' in source
    assert 'stage_d_manifest.get("status") != str(' in source
    assert 'inference_cfg["selector_predict_base_row_chunk_size"]' in source
    assert 'inference_cfg["signed_top1_value_parity_atol"]' in source
    assert 'config["model"]["selector"]' not in source
    assert 'config["guards"]["stage_s"]' not in source
    assert "kaggle competitions submit" not in source
    assert "__file__" not in source
    assert source.count("# %% [markdown]") >= 10

    matrix_start = source.index("matrix_frame = pd.concat(")
    matrix_end = source.index("if list(matrix_frame.columns)", matrix_start)
    matrix_block = source[matrix_start:matrix_end]
    assert matrix_block.index("test_frame[base_feature_columns]") < (
        matrix_block.index("aligned_compact[parent_compact_features]")
    )
    assert matrix_block.index("aligned_compact[parent_compact_features]") < (
        matrix_block.index("formation_surface[formation_features]")
    )
    assert matrix_block.index("formation_surface[formation_features]") < (
        matrix_block.index("aligned_signed[signed_compact_features]")
    )


def test_inference_raw_context_contract_is_available_on_public_test() -> None:
    config = load_config()
    allowlist = config["features"]["raw_context"]["horizontal_numeric_allowlist"]
    files = sorted(Path("data/raw/test").glob("*__horizontal_well.csv"))

    assert len(files) == 3
    for path in files:
        columns = pd.read_csv(path, nrows=0).columns.tolist()
        assert all(column in columns for column in allowlist)


def test_frozen_manifest_and_schema_hashes_match_saved_metadata() -> None:
    config = load_config()
    data = config["data"]
    paths = {
        "exp264_nested_compact_manifest_sha256": Path(
            "experiments/exp264_exp263_candidate_confidence_dual_selector/"
            "kaggle/output/stage_c_v6/artifacts/nested_compact_manifest.json"
        ),
        "exp264_compact_schema_file_sha256": Path(
            "experiments/exp264_exp263_candidate_confidence_dual_selector/"
            "kaggle/output/stage_c_v6/artifacts/compact_meta_schema.json"
        ),
        "exp335_signed_compact_manifest_sha256": Path(
            "experiments/exp335_signed_residual_meta_on_exp264/"
            "kaggle/output/stage_s_v3/artifacts/signed_compact_manifest.json"
        ),
        "exp335_signed_compact_schema_file_sha256": Path(
            "experiments/exp335_signed_residual_meta_on_exp264/"
            "kaggle/output/preflight_v2/artifacts/signed_compact_schema.json"
        ),
    }
    require_saved_files(*paths.values())
    for field, path in paths.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == data[field]


def test_key_columns_expected_by_alignment_remain_frozen() -> None:
    assert KEY_COLUMNS == ["id", "well", "well_row_idx", "outer_fold", "md_since"]
