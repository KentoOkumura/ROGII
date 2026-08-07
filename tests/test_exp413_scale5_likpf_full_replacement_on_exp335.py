from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.candidate_selector_pipeline import candidate_ids, contract_by_id, sha256_file
from src.likpf_full_replacement import (
    CHANGED_CANDIDATES,
    REPLACEMENT_VALUE_SOURCE,
    UNCHANGED_CANDIDATES,
    ReplacementCandidateCache,
    _load_module,
    _assert_replacement_alignment,
    _formula_parity_max_abs,
    build_bank_from_primitives,
    evaluate_replacement_gate,
    patch_base_replay_primitive,
    replacement_cost_contract,
    require_stage_authorization,
    validate_replacement_contract,
    verify_replacement_stage_0_root,
)

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "exp413_scale5_likpf_full_replacement_on_exp335"
PARENT = ROOT / "experiments" / "exp264_exp263_candidate_confidence_dual_selector"


def load_config() -> dict:
    return yaml.safe_load((EXP / "config.yaml").read_text())


def load_contract() -> dict:
    return yaml.safe_load((PARENT / "candidate_contract.yaml").read_text())


def test_dynamic_loader_registers_module_before_dataclass_execution(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "dynamic_dataclass_module.py"
    module_path.write_text(
        "from __future__ import annotations\n"
        "from dataclasses import dataclass\n"
        "@dataclass(frozen=True)\n"
        "class CandidateSpec:\n"
        "    name: str\n"
    )
    module_name = "exp413_test_dynamic_dataclass_module"
    sys.modules.pop(module_name, None)
    module = _load_module(module_path, module_name)
    try:
        assert module.CandidateSpec("likpf").name == "likpf"
        assert sys.modules[module_name] is module
    finally:
        sys.modules.pop(module_name, None)


def test_config_records_stage_d_and_authorizes_kaggle_submission_output() -> None:
    config = load_config()
    contract = load_contract()
    evidence = validate_replacement_contract(config, contract)
    cost = replacement_cost_contract(config)
    assert config["experiment"]["status"] in {
        "kaggle_submission_output_v3_preparing",
        "kaggle_submission_output_v3_complete_submit_check_pass",
        "code_submission_v3_hidden_rerun_failed_v4_preparing",
        "hidden_compatible_submission_output_v4_complete_submit_check_pass",
        "code_submission_v4_complete_public_lb_7p201",
    }
    assert config["authorization"]["implementation_approved"] is True
    assert config["authorization"]["canonical_notebook_adoption_approved"] is False
    assert config["authorization"]["kaggle_package_approved"] is True
    assert config["authorization"]["stage_0_run_approved"] is True
    assert config["authorization"]["selector_train_approved"] is True
    assert config["authorization"]["signed_selector_train_approved"] is True
    assert config["authorization"]["stage_s_kaggle_package_approved"] is True
    assert config["authorization"]["downstream_gpu_train_approved"] is True
    assert config["authorization"]["stage_d_kaggle_package_approved"] is True
    assert config["authorization"]["inference_implementation_approved"] is True
    assert config["authorization"]["inference_run_approved"] is True
    assert config["authorization"]["submission_file_generation_approved"] is True
    assert config["authorization"]["external_submission_approved"] is False
    assert config["implementation"]["inference_enabled"] is True
    assert config["implementation"]["submission_enabled"] is True
    inference = config["inference"]
    assert (
        inference["status"]
        == "user_authorized_2026_07_29_kaggle_submission_output"
    )
    assert inference["stage_d_primary_gate_passed"] is True
    assert inference["runtime"] == "kaggle_cpu"
    assert inference["booster_training_count"] == 0
    assert isinstance(inference["run_enabled"], bool)
    assert inference["generate_submission_file"] is True
    assert inference["submit_to_kaggle"] is False
    assert inference["competition_submit_authorized"] is False
    assert inference["parent_selector_model_count"] == 40
    assert inference["signed_selector_model_count"] == 20
    assert inference["tvt_model_count"] == 15
    assert inference["expected_final_feature_count"] == 370
    assert config["execution"]["run_flags"] == {
        "replacement_preflight": False,
        "nested_selector_train": False,
        "signed_selector_train": False,
        "downstream_gpu_train": False,
    }
    assert evidence["changed_candidates"] == list(CHANGED_CANDIDATES)
    assert evidence["unchanged_candidates"] == list(UNCHANGED_CANDIDATES)
    assert config["data"]["exp072_train_feature_cache"][
        "expected_likpf_dependency_columns"
    ] == ["likpf_mean_d"]
    assert config["replacement"]["feature_graph"]["clean_base"][
        "expected_likpf_named_feature_count"
    ] == 22
    assert config["replacement"]["parent_old_mean_parity_max_abs_ft"] == 0.001
    assert cost == {
        "replacement_variants": 1,
        "cpu_selector_boosters": 40,
        "cpu_signed_selector_boosters": 20,
        "gpu_downstream_boosters": 15,
        "total_boosters": 75,
        "parent_control_retraining_boosters": 0,
        "train_pf_well_runs": 0,
        "hmm_well_runs": 0,
        "beam_well_runs": 0,
    }
    completed = config["execution"]["completed_stages"]["replacement_preflight"]
    assert completed["kernel_version"] == 3
    assert completed["models_trained"] == 0
    assert completed["pf_well_runs"] == 0
    completed_stage_c = config["execution"]["completed_stages"][
        "nested_selector_train"
    ]
    assert completed_stage_c["kernel_version"] == 3
    assert completed_stage_c["technical_pass"] is True
    assert completed_stage_c["score_guard_pass"] is True
    assert completed_stage_c["leakage_audit_pass"] is True
    assert completed_stage_c["models_trained"] == 40
    assert completed_stage_c["compact_partitions"] == 25
    assert completed_stage_c["compact_rows"] == 18_919_945
    assert completed_stage_c["outer_valid_score_long_rows"] == 45_407_868
    completed_stage_s = config["execution"]["completed_stages"][
        "signed_selector_train"
    ]
    assert completed_stage_s["kernel_version"] == 1
    assert completed_stage_s["technical_pass"] is True
    assert completed_stage_s["score_gate_pass"] is True
    assert completed_stage_s["stage_s_gate_passed"] is True
    assert completed_stage_s["models_trained"] == 20
    assert completed_stage_s["compact_partitions"] == 25
    assert completed_stage_s["compact_rows"] == 18_919_945
    assert completed_stage_s["outer_valid_score_long_rows"] == 45_407_868
    assert completed_stage_s["improved_outer_folds"] == 5
    assert completed_stage_s["improved_candidates"] == 11
    assert completed_stage_s["control_models_retrained"] == 0
    completed_stage_d = config["execution"]["completed_stages"][
        "downstream_gpu_train"
    ]
    assert completed_stage_d["kernel_version"] == 2
    assert completed_stage_d["technical_pass"] is True
    assert completed_stage_d["primary_gate_pass"] is True
    assert completed_stage_d["models_trained"] == 15
    assert completed_stage_d["unique_model_sha_count"] == 15
    assert completed_stage_d["replacement_rmse"] == pytest.approx(
        7.884802794404715
    )
    assert completed_stage_d["gain_ft"] == pytest.approx(0.26130496147630744)
    assert completed_stage_d["nonworse_folds"] == 5
    assert completed_stage_d["maximum_scope_delta_rmse_ft"] < 0.02
    assert completed_stage_d["final_feature_count"] == 370
    assert completed_stage_d["control_models_retrained"] == 0
    assert completed_stage_d["pf_well_runs"] == 0
    assert completed_stage_d["inference_executed"] is True
    assert completed_stage_d["submission_generated"] is False
    completed_inference = config["execution"]["completed_stages"][
        "current_test_inference"
    ]
    assert completed_inference["kernel_version"] == 4
    assert completed_inference["output_validation_status"] == "pass"
    assert completed_inference["rows"] == 14_151
    assert completed_inference["wells"] == 3
    assert completed_inference["parent_selector_model_count"] == 40
    assert completed_inference["signed_selector_model_count"] == 20
    assert completed_inference["tvt_model_count"] == 15
    assert completed_inference["booster_training_count"] == 0
    assert completed_inference["submission_file_generated"] is True
    assert completed_inference["external_submission_performed"] is False
    submission = config["submission_file"]
    assert submission["status"] in {
        "kaggle_v3_submission_output_authorized_preparing",
        "kaggle_v3_submission_output_complete_submit_check_pass",
        "hidden_compatible_kaggle_v4_submission_output_complete_submit_check_pass",
    }
    assert submission["kaggle_kernel_version"] == 4
    assert isinstance(submission["kaggle_generated"], bool)
    assert submission["expected_columns"] == ["id", "tvt"]
    local_preflight = submission["local_preflight"]
    assert local_preflight["status"] == "pass_not_a_kaggle_output"
    assert local_preflight["rows"] == 14_151
    assert local_preflight["unique_ids"] == 14_151
    assert local_preflight["missing_values"] == 0
    assert local_preflight["nonfinite_values"] == 0
    assert local_preflight["source_value_parity"] == "exact"
    assert local_preflight["submit_check_status"] == "pass"
    if submission["kaggle_generated"]:
        assert submission["rows"] == 14_151
        assert submission["unique_ids"] == 14_151
        assert submission["missing_values"] == 0
        assert submission["nonfinite_values"] == 0
        assert submission["source_value_parity"] == "exact"
        assert submission["kaggle_submit_check_status"] == "pass"
    assert submission["external_submission_authorized"] is False
    assert submission["external_submission_performed"] is False
    failed_stage_c = config["execution"]["failed_runs"]["nested_selector_train"][0]
    assert failed_stage_c["kernel_version"] == 1
    failed_stage_c_v2 = config["execution"]["failed_runs"]["nested_selector_train"][1]
    assert failed_stage_c_v2["kernel_version"] == 2
    assert failed_stage_c_v2["completed_booster_logs"] == 0
    assert failed_stage_c["completed_booster_logs"] == 40
    assert failed_stage_c["recoverable_output_files"] == 0
    failed_inference_v1 = config["execution"]["failed_runs"][
        "current_test_inference"
    ][0]
    assert failed_inference_v1["kernel_version"] == 1
    assert failed_inference_v1["completed_saved_model_predictions"] == 0
    assert failed_inference_v1["recoverable_prediction_files"] == 0
    with pytest.raises(RuntimeError, match="replacement_preflight"):
        require_stage_authorization(config, "replacement_preflight")
    with pytest.raises(RuntimeError, match="nested_selector_train"):
        require_stage_authorization(config, "nested_selector_train")
    with pytest.raises(RuntimeError, match="signed_selector_train"):
        require_stage_authorization(config, "signed_selector_train")
    with pytest.raises(RuntimeError, match="downstream_gpu_train"):
        require_stage_authorization(config, "downstream_gpu_train")
    assert config["model"]["downstream_tvt"]["lightgbm_config_indices"] == [0, 1, 2]
    assert config["model"]["downstream_tvt"]["folds"] == 5
    assert config["model"]["downstream_tvt"]["planned_gpu_boosters"] == 15
    assert config["model"]["downstream_tvt"]["control_retraining_boosters"] == 0
    assert config["model"]["downstream_tvt"]["log_evaluation_period"] == 100
    assert config["model"]["downstream_tvt"]["matrix_copy_chunk_columns"] == 32
    assert config["runtime"]["kaggle"]["downstream_gpu_train"] == {
        "enable_gpu": True,
        "machine_shape": "NvidiaTeslaT4",
    }
    assert config["runtime"]["kaggle"]["current_test_inference"] == {
        "enable_gpu": False,
        "machine_shape": "",
    }


def test_alignment_is_exact_at_parent_float32_cache_precision() -> None:
    parent = pd.DataFrame(
        {
            "id": ["row_0"],
            "well": ["well_a"],
            "well_row_idx": np.asarray([10], dtype=np.int32),
            "last_known_tvt": np.asarray([8249.673], dtype=np.float32),
            "md_since": np.asarray([42.2], dtype=np.float32),
        }
    )
    replacement = pd.DataFrame(
        {
            "id": ["row_0"],
            "well_id": ["well_a"],
            "row_idx": np.asarray([10], dtype=np.int64),
            "last_known_tvt": parent["last_known_tvt"].astype(np.float64)
            + np.float64(0.00046875),
            "md_since": parent["md_since"].astype(np.float64),
        }
    )
    assert np.array_equal(
        parent["last_known_tvt"].to_numpy(np.float32),
        replacement["last_known_tvt"].to_numpy(np.float32),
    )
    _assert_replacement_alignment(parent, replacement)

    replacement.loc[0, "last_known_tvt"] += 0.01
    with pytest.raises(ValueError, match="float32 cache precision"):
        _assert_replacement_alignment(parent, replacement)


def _write_primitive_partition(
    root: Path,
    name: str,
    values: np.ndarray,
    *,
    confidence_source: str,
) -> None:
    rows = len(values)
    base = pd.DataFrame(
        {
            "id": [f"row_{index}" for index in range(rows)],
            "well": ["well_a"] * rows,
            "well_row_idx": np.arange(10, 10 + rows, dtype=np.int32),
            "outer_fold": np.zeros(rows, dtype=np.int8),
            "md_since": np.arange(1, rows + 1, dtype=np.float32),
        }
    )
    value = base.copy()
    value["last_known_tvt"] = np.float32(1000.0)
    value["candidate_id"] = name
    value["candidate_tvt"] = np.asarray(values, dtype=np.float32)
    value["candidate_available"] = True
    value_path = root / "candidate_values" / name / "fold=0" / "part-000.parquet"
    value_path.parent.mkdir(parents=True, exist_ok=True)
    value.to_parquet(value_path, index=False)

    confidence = base.copy()
    confidence["candidate_id"] = name
    confidence["confidence_source"] = confidence_source
    confidence["confidence_valid"] = True
    confidence["confidence_missing_fields"] = ""
    confidence_path = (
        root / "candidate_confidence" / name / "fold=0" / "part-000.parquet"
    )
    confidence_path.parent.mkdir(parents=True, exist_ok=True)
    confidence.to_parquet(confidence_path, index=False)


def test_overlay_cache_changes_exactly_five_candidates_and_never_reads_old_mean(
    tmp_path: Path,
) -> None:
    contract = load_contract()
    ids = candidate_ids(contract)
    specs = contract_by_id(contract)
    parent_root = tmp_path / "parent"
    replacement_root = tmp_path / "replacement"
    primitive_ids = [name for name in ids if specs[name]["kind"] == "primitive"]
    old_primitives: dict[str, np.ndarray] = {}
    for position, name in enumerate(primitive_ids):
        values = np.arange(3, dtype=np.float32) + np.float32(1000 + position * 10)
        old_primitives[name] = values
        _write_primitive_partition(
            parent_root,
            name,
            values,
            confidence_source=f"parent_{name}",
        )
    replacement_values = old_primitives["likpf_mean"] + np.float32(7.0)
    _write_primitive_partition(
        replacement_root,
        "likpf_mean",
        replacement_values,
        confidence_source=REPLACEMENT_VALUE_SOURCE,
    )

    cache = ReplacementCandidateCache(parent_root, contract, replacement_root)
    bundle = cache.load_fold(0)
    new_primitives = dict(old_primitives)
    new_primitives["likpf_mean"] = replacement_values
    expected_new = build_bank_from_primitives(new_primitives, contract)
    expected_old = build_bank_from_primitives(old_primitives, contract)
    np.testing.assert_array_equal(bundle.values, expected_new)
    assert (
        _formula_parity_max_abs(bundle.values, contract, CHANGED_CANDIDATES[1:])
        == 0.0
    )
    broken = bundle.values.copy()
    broken[:, ids.index(CHANGED_CANDIDATES[1])] += np.float32(0.25)
    assert _formula_parity_max_abs(broken, contract, CHANGED_CANDIDATES[1:]) > 0.0
    changed = [
        name
        for position, name in enumerate(ids)
        if np.any(bundle.values[:, position] != expected_old[:, position])
    ]
    unchanged = [
        name
        for position, name in enumerate(ids)
        if np.array_equal(bundle.values[:, position], expected_old[:, position])
    ]
    assert changed == list(CHANGED_CANDIDATES)
    assert unchanged == list(UNCHANGED_CANDIDATES)
    assert (
        bundle.confidence["likpf_mean"]["confidence_source"].unique().tolist()
        == [REPLACEMENT_VALUE_SOURCE]
    )
    assert not any(
        "parent_likpf_mean" in str(value)
        for value in bundle.confidence["likpf_mean"].to_numpy().reshape(-1)
    )


def test_base_replay_is_reloaded_then_primitive_is_replaced_at_source() -> None:
    base = pd.DataFrame(
        {
            "id": ["a_10", "a_11", "b_4"],
            "well": ["a", "a", "b"],
            "last_known_tvt": np.asarray([1000, 1000, 2000], dtype=np.float32),
            "md_since": np.asarray([1, 2, 1], dtype=np.float32),
            "likpf_mean_d": np.asarray([1, 2, 3], dtype=np.float32),
            "other_feature": np.asarray([5, 6, 7], dtype=np.float32),
        }
    )
    frozen = pd.DataFrame(
        {
            "id": ["b_4", "a_10", "a_11"],
            "well_id": ["b", "a", "a"],
            "row_idx": [4, 10, 11],
            "last_known_tvt": [2000.0, 1000.0, 1000.0],
            "md_since": [1.0, 1.0, 2.0],
            REPLACEMENT_VALUE_SOURCE: [2010.0, 1004.0, 1008.0],
            "likpf_mean_x1p0": [2003.0, 1001.0, 1002.0],
        }
    )
    rebuilt, evidence = patch_base_replay_primitive(base, frozen)
    np.testing.assert_array_equal(
        rebuilt["likpf_mean_d"].to_numpy(np.float32),
        np.asarray([4, 8, 10], dtype=np.float32),
    )
    np.testing.assert_array_equal(rebuilt["other_feature"], base["other_feature"])
    assert evidence["changed_rows"] == 3
    assert evidence["old_mean_retained_in_output"] is False
    assert "likpf_mean_x1p0" not in rebuilt


def test_tail_readouts_are_mandatory_but_report_only(tmp_path: Path) -> None:
    config = copy.deepcopy(load_config())
    config["validation"]["promotion"].update(
        {
            "minimum_pooled_rmse_gain_ft": -100.0,
            "minimum_nonworse_folds": 0,
            "maximum_scope_delta_rmse_ft": 100.0,
        }
    )
    rows = 10
    base = pd.DataFrame(
        {
            "id": [f"row_{index}" for index in range(rows)],
            "well": [f"well_{index}" for index in range(rows)],
            "last_known_tvt": np.zeros(rows, dtype=np.float32),
            "target": np.zeros(rows, dtype=np.float32),
            "md_since": np.asarray(
                [10, 20, 300, 400, 1100, 1200, 30, 500, 1300, 40],
                dtype=np.float32,
            ),
        }
    )
    parent = base[["id", "well"]].copy()
    parent["outer_fold"] = np.arange(rows, dtype=np.int8) % 5
    parent["signed_residual_meta_addonly__lgb_mean__pred_tvt"] = np.float32(1.0)
    new_prediction = np.ones(rows, dtype=np.float32)
    new_prediction[0] = np.float32(8.0)
    assignment = pd.DataFrame(
        {
            "well_id": base["well"],
            "verification_like_spatial_role": ["valid"] * rows,
            "verification_like_typewell_purged_role": ["valid"] * rows,
        }
    )
    assignment_path = tmp_path / "assignment.csv"
    assignment.to_csv(assignment_path, index=False)
    gate, *_ = evaluate_replacement_gate(
        config=config,
        base_frame=base,
        saved_parent=parent,
        oof_fold=parent["outer_fold"].to_numpy(np.int8),
        new_prediction=new_prediction,
        hidden_like_assignment_path=assignment_path,
        technical_checks={"synthetic": True},
    )
    assert gate["tail_readout"]["worst_well_delta_rmse"] > 1.0
    assert gate["tail_readout"]["policy"] == "report_only_not_automatic_stop"
    assert gate["passed"] is True


def test_compact_jupytext_candidate_is_readable_and_canonical_remains_placeholder() -> None:
    source = (
        EXP
        / "exp413_scale5_likpf_full_replacement_on_exp335_compact_selfcontained_train.py"
    ).read_text()
    assert "run_replacement_preflight(" in source
    assert "run_stage_c(" in source
    assert "run_stage_s(" in source
    assert "run_replacement_stage_d(" in source
    assert "verify_replacement_stage_0_root(" in source
    assert "verify_replacement_stage_c_root(" in source
    assert "verify_replacement_stage_s_root(" in source
    assert 'output_dir / "reproducibility_manifest.json"' in source
    assert '"status": "stage_c_inputs_frozen"' in source
    assert source.index('"status": "stage_c_inputs_frozen"') < source.index(
        "stage_summary = run_stage_c("
    )
    assert "saved_exp335_control_retraining" in source
    assert "Inference executed: False" in source
    assert "Submission generated or submitted: False" in source
    assert "__file__" not in source
    assert source.count("# %% [markdown]") >= 10
    assert not (EXP / "exp413_scale5_likpf_full_replacement_on_exp335_train.py").exists()


def test_current_test_inference_is_full_replacement_and_writes_submission() -> None:
    source = (
        EXP
        / "exp413_scale5_likpf_full_replacement_on_exp335_current_test_inference.py"
    ).read_text()
    canonical = (
        EXP / "exp413_scale5_likpf_full_replacement_on_exp335_inference.ipynb"
    )
    config = load_config()
    inference = config["inference"]
    assert canonical.exists()
    assert "build_replay_test_frame()" in source
    assert 'pf_frame["likpf_mean"] = scale5' in source
    assert (
        'pf_frame["likpf_mean_d"] = '
        'pf_frame["likpf_scale_5_d"].to_numpy(np.float32)'
    ) in source
    assert "replay_stable_seed(\"likpf\", \"test\", well)" in source
    assert (
        'parent_config["guards"]["stage_s"]["saved_top1_value_parity_atol"]'
        in source
    )
    assert 'float(\n                config["guards"]["stage_s"]' not in source
    assert '"temperature": 5.0' in source
    assert '"booster_training_count": 0' in source
    assert '"submission_file_generated": True' in source
    assert "predictions.to_csv(prediction_path" in source
    assert "submission.to_csv(paths.submission_path, index=False)" in source
    assert 'len(pf_frame) != len(sample)' in source
    assert 'set(pf_frame["id"]) != set(sample["id"])' in source
    assert "current_test_expected_rows" not in source
    assert "current_test_expected_wells" not in source
    assert "kaggle competitions submit" not in source
    assert "__file__" not in source
    assert inference["nested_selector_model_manifest_sha256"] == (
        config["execution"]["completed_stages"]["nested_selector_train"][
            "model_manifest_sha256"
        ]
    )
    assert inference["signed_selector_model_manifest_sha256"] == (
        config["execution"]["completed_stages"]["signed_selector_train"][
            "model_manifest_sha256"
        ]
    )
    assert inference["tvt_model_manifest_sha256"] == (
        config["execution"]["completed_stages"]["downstream_gpu_train"][
            "model_manifest_sha256"
        ]
    )


def test_clean273_dependency_and_source_sha_contracts_are_frozen() -> None:
    config = load_config()
    schema = pd.read_csv(
        ROOT
        / "experiments"
        / "exp072_exp063_full_replay_feature_cache"
        / "artifacts"
        / "exp063_full_replay_feature_cache_feature_schema.csv"
    )
    base_likpf = [
        feature for feature in schema["feature"].astype(str) if "likpf" in feature.lower()
    ]
    assert base_likpf == config["data"]["exp072_train_feature_cache"][
        "expected_likpf_dependency_columns"
    ]
    allowlist = pd.read_csv(
        PARENT / "artifacts" / "feature_availability_audit" / "exp218_clean_273_allowlist.csv"
    )
    assert (
        allowlist["feature"].astype(str).str.contains("likpf", case=False).sum()
        == config["data"]["clean_base_allowlist"]["expected_likpf_named_feature_count"]
        == 22
    )
    source_specs = (
        (
            ROOT
            / "experiments"
            / "exp218_gr_wavelet_rotation_confidence_features_on_exp148"
            / "gr_wavelet_rotation_confidence_features_on_exp148.py",
            config["data"]["exp218_source"]["script_sha256"],
        ),
        (
            ROOT
            / "experiments"
            / "exp145_learned_likelihood_rawtest_feature_generator_parity"
            / "learned_likelihood_rawtest_feature_generator_parity.py",
            config["data"]["exp145_source"]["script_sha256"],
        ),
        (
            ROOT
            / "experiments"
            / "exp145_learned_likelihood_rawtest_feature_generator_parity"
            / "pf_multi_observation_likelihood_probe.py",
            config["data"]["exp145_source"]["multiobs_script_sha256"],
        ),
        (
            ROOT
            / "experiments"
            / "exp111_learned_pf_observation_likelihood_probe"
            / "kaggle"
            / "output"
            / "train_v1"
            / "artifacts"
            / "exp111_learned_pf_observation_likelihood_probe_model_manifest.json",
            config["data"]["exp111_saved_models"]["manifest_sha256"],
        ),
    )
    for path, expected_sha in source_specs:
        assert sha256_file(path) == expected_sha


def test_stage_0_lineage_verifier_rejects_partition_tampering(tmp_path: Path) -> None:
    config = copy.deepcopy(load_config())
    config["validation"]["expected_rows"] = 3
    selector_schema = {
        "features": [f"selector_{index}" for index in range(88)],
        "feature_schema_sha256": "selector-logical",
    }
    selector_schema_path = tmp_path / "feature_schema.json"
    selector_schema_path.write_text(json.dumps(selector_schema))
    selector_catalog_path = tmp_path / "feature_catalog.csv"
    selector_catalog_path.write_text("feature,status\nselector_0,selected\n")
    compact_schema_path = tmp_path / "compact_meta_schema.json"
    compact_schema_path.write_text(
        json.dumps({"features": [f"compact_{index}" for index in range(74)]})
    )
    selector_contract = config["data"]["selector_contract"]
    selector_contract["feature_schema_file_sha256"] = sha256_file(selector_schema_path)
    selector_contract["feature_schema_logical_sha256"] = "selector-logical"
    selector_contract["feature_catalog_sha256"] = sha256_file(selector_catalog_path)

    partition_rows = []
    for fold in range(5):
        value_path = (
            tmp_path
            / "replacement_candidate_cache"
            / "candidate_values"
            / "likpf_mean"
            / f"fold={fold}"
            / "part-000.parquet"
        )
        confidence_path = (
            tmp_path
            / "replacement_candidate_cache"
            / "candidate_confidence"
            / "likpf_mean"
            / f"fold={fold}"
            / "part-000.parquet"
        )
        value_path.parent.mkdir(parents=True, exist_ok=True)
        confidence_path.parent.mkdir(parents=True, exist_ok=True)
        value_path.write_bytes(f"value-{fold}".encode())
        confidence_path.write_bytes(f"confidence-{fold}".encode())
        partition_rows.append(
            {
                "fold": fold,
                "rows": 3 if fold == 0 else 0,
                "value_path": str(
                    value_path.relative_to(
                        tmp_path / "replacement_candidate_cache"
                    )
                ),
                "value_file_sha256": sha256_file(value_path),
                "confidence_path": str(
                    confidence_path.relative_to(
                        tmp_path / "replacement_candidate_cache"
                    )
                ),
                "confidence_file_sha256": sha256_file(confidence_path),
            }
        )
    semantic = {
        "semantic_slot": "likpf_mean",
        "value_source": REPLACEMENT_VALUE_SOURCE,
        "old_mean_in_candidate_or_model_input": False,
        "changed_candidates": list(CHANGED_CANDIDATES),
        "unchanged_candidates": list(UNCHANGED_CANDIDATES),
        "partition_count": 5,
        "partitions": partition_rows,
    }
    semantic_path = tmp_path / "replacement_semantic_manifest.json"
    semantic_path.write_text(json.dumps(semantic))
    frozen_spec = config["data"]["exp404_scale5_train_prediction"]
    preflight = {
        "passed": True,
        "replacement_semantic_manifest_sha256": sha256_file(semantic_path),
        "frozen_prediction": {
            "raw_sha256": frozen_spec["expected_raw_sha256"],
            "decompressed_sha256": frozen_spec["expected_decompressed_sha256"],
            "logical_sha256": frozen_spec["expected_logical_sha256"],
            "schema_sha256": frozen_spec["expected_schema_sha256"],
        },
    }
    (tmp_path / "replacement_preflight.json").write_text(json.dumps(preflight))
    evidence = verify_replacement_stage_0_root(tmp_path, config)
    assert evidence["rows"] == 3
    assert evidence["partition_count"] == 5

    tampered = (
        tmp_path
        / "replacement_candidate_cache"
        / "candidate_values"
        / "likpf_mean"
        / "fold=0"
        / "part-000.parquet"
    )
    tampered.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="partition SHA mismatch"):
        verify_replacement_stage_0_root(tmp_path, config)
