from __future__ import annotations

import copy
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from tests.test_support import require_saved_files

ROOT = Path(__file__).resolve().parents[3]
EXP_NAME = "exp396_fold_safe_exp111_score_27_addonly_on_exp287"
EXP_DIR = ROOT / "experiments" / EXP_NAME
TRAIN = EXP_DIR / f"{EXP_NAME}_compact_selfcontained_train.py"
INFERENCE = EXP_DIR / f"{EXP_NAME}_compact_selfcontained_inference.py"
CONFIG = EXP_DIR / "config.yaml"


def load_module(name: str, path: Path, env_name: str):
    previous = os.environ.get(env_name)
    os.environ[env_name] = "1"
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        if previous is None:
            os.environ.pop(env_name, None)
        else:
            os.environ[env_name] = previous


@pytest.fixture(scope="module")
def module():
    return load_module("exp396_train_test", TRAIN, "EXP396_IMPORT_ONLY")


@pytest.fixture(scope="module")
def inference_module():
    return load_module(
        "exp396_inference_test",
        INFERENCE,
        "EXP396_INFERENCE_IMPORT_ONLY",
    )


@pytest.fixture()
def config() -> dict:
    value = yaml.safe_load(CONFIG.read_text())
    assert isinstance(value, dict)
    return value


def make_target_free_source(module, config, rows: int = 12) -> pd.DataFrame:
    index = np.arange(rows, dtype=np.float32)
    last = 100.0 + index
    data: dict[str, np.ndarray | list[str]] = {
        "id": [f"id_{value:03d}" for value in range(rows)],
        "well": [f"well_{value // 2:03d}" for value in range(rows)],
        "last_known_tvt": last,
        "pf_ancc_std": 0.1 + index,
        "beam_mean_d": 2.0 + index * 0.1,
        "sc_ens_d": 3.0 + index * 0.1,
        "hyb_d": 4.0 + index * 0.1,
        "eval_len": 50.0 + index,
        "md_since": 10.0 + index,
        "likpf_mean_d": 5.0 + index * 0.1,
        "multiobs_score_max": 0.9 + index * 0.001,
        "multiobs_score_mean": 0.5 + index * 0.001,
        "multiobs_score_gap": 0.4 + index * 0.001,
        "multiobs_top1_source_id": index % 5,
        "multiobs_top1_mae": 1.0 + index * 0.01,
        "multiobs_top1_ncc": 0.8 + index * 0.001,
    }
    offsets = [1.0, 3.0, 5.0, 7.0, 9.0]
    for candidate_index, candidate in enumerate(module.CANDIDATE_ORDER):
        data[candidate] = last + offsets[candidate_index]
        data[f"multiobs_score_{candidate}"] = (
            1.0 - 0.1 * candidate_index + index * 0.001
        )
        data[f"multiobs_mae_{candidate}"] = (
            0.5 + 0.2 * candidate_index + index * 0.001
        )
        data[f"multiobs_ncc_{candidate}"] = (
            0.9 - 0.05 * candidate_index + index * 0.001
        )
    return pd.DataFrame(data)


def test_stage_b_execution_contract_is_fail_closed(module, config) -> None:
    report = module.validate_scientific_contract(config)
    assert report["stage"] == "implementation_only"
    assert config["execution"]["run_train"] is False
    assert config["execution"]["kaggle_push_approved"] is False
    assert config["outcome"]["stage_b_gate_passed"] is False
    with pytest.raises(
        RuntimeError,
        match="selected stage does not execute an approved stage",
    ):
        module.validate_scientific_contract(
            config,
            require_execution_approval=True,
        )
    assert report["planned_cpu_boosters"] == 40
    assert report["stage_b_planned_gpu_boosters"] == 15
    assert report["control_retraining_boosters"] == 0
    assert report["inference"] is False
    assert report["submission"] is False
    assert config["execution"]["implementation_approval_source"] == (
        "user_message_implement_exp396_2026_07_25"
    )
    assert config["execution"]["stage_b_gpu_run_approval_source"] == (
        "user_message_execute_stage_b_2026_07_25"
    )

    authorized = copy.deepcopy(config)
    authorized["execution"]["stage"] = "stage_b_gpu_tvt_train"
    authorized["execution"]["kaggle_push_approved"] = True
    authorized["execution"]["run_train"] = True
    approved = module.validate_scientific_contract(
        authorized,
        require_execution_approval=True,
    )
    assert approved["stage"] == "stage_b_gpu_tvt_train"

    unauthorized = copy.deepcopy(authorized)
    unauthorized["execution"]["stage"] = "stage_a_preflight_only"
    unauthorized["execution"]["stage_a_preflight_run_approved"] = False
    with pytest.raises(RuntimeError, match="preflight requires separate user approval"):
        module.validate_scientific_contract(
            unauthorized,
            require_execution_approval=True,
        )
    unauthorized["execution"]["stage"] = "stage_a_cpu_scorer_train"
    unauthorized["execution"]["stage_a_cpu_run_approved"] = False
    with pytest.raises(RuntimeError, match="40 CPU boosters require separate"):
        module.validate_scientific_contract(
            unauthorized,
            require_execution_approval=True,
        )
    unauthorized["execution"]["stage"] = "stage_b_preflight_only"
    unauthorized["execution"]["stage_b_implementation_approved"] = False
    with pytest.raises(RuntimeError, match="Stage B is not approved"):
        module.validate_scientific_contract(unauthorized)
    unauthorized = copy.deepcopy(authorized)
    unauthorized["execution"]["stage_b_gpu_run_approved"] = False
    with pytest.raises(RuntimeError, match="15 GPU boosters require separate"):
        module.validate_scientific_contract(
            unauthorized,
            require_execution_approval=True,
        )


def test_fixed_schemas_and_hashes_are_locked(module, config) -> None:
    row_features = module.fixed_row_feature_columns(config)
    assert len(row_features) == 32
    assert len(module.model_feature_columns(row_features)) == 48
    assert len(module.SCORE_CORE_COLUMNS) == 10
    assert len(module.FEATURE_COLUMNS_27) == 27
    assert module.validate_derived_schema_hashes(row_features, config) == {
        "model_feature_schema_sha256": (
            "2168c04d71c1bb1de9cdeddfd24fcc83f14d43d269438645af4e9cc3d5e4fe4e"
        ),
        "score_core_schema_sha256": (
            "8ea2fc2161b6b9fa142c9fd12a26a57e9ec259bd131550aba28c617919c82df7"
        ),
        "derived_27_schema_sha256": (
            "1d2c4bd40944abffdbbc4f9bddfbff4efe23d06c3e053069bef05d51ea8a5f44"
        ),
    }
    assert config["guards"]["dependent_grwr_six"] == module.DEPENDENT_GRWR_SIX
    assert config["model"]["source_surface"]["final_feature_schema_sha256"] == (
        "07f6c2b51d166f210bae18720c32fae638aead255b24adda4ac598eaac517630"
    )
    reference = (
        ROOT
        / "experiments"
        / "exp111_learned_pf_observation_likelihood_probe"
    )
    feature_schema = (
        reference
        / "kaggle/output/train_v1/artifacts/"
        "exp111_learned_pf_observation_likelihood_probe_feature_schema.csv"
    )
    model_manifest = (
        reference
        / "kaggle/output/train_v1/artifacts/"
        "exp111_learned_pf_observation_likelihood_probe_model_manifest.json"
    )
    require_saved_files(feature_schema, model_manifest)
    evidence = module.verify_exp111_contract_files(
        reference / "learned_pf_observation_likelihood_probe.py",
        reference / "config.yaml",
        feature_schema,
        model_manifest,
        config,
    )
    assert evidence["saved_model_prediction_use"] is False
    assert evidence["model_manifest_sha256"] == (
        "178e8b3124b817a2b230080fc041aaaee1b06941e5a4223a68cc31bf26e68010"
    )


def test_target_free_feature_builder_and_candidate_major_labels(
    module,
    config,
) -> None:
    source = make_target_free_source(module, config)
    specs = module.candidate_specs(config)
    frame, row_features, candidate_values = module.prepare_target_free_wide_features(
        source,
        specs,
        configured_row_columns=config["scorer"]["exp111_row_context_columns"],
        configured_global_columns=config["scorer"]["exp111_multiobs_global_columns"],
    )
    long = module.build_target_free_candidate_long(
        frame,
        np.asarray([0, 2, 4], dtype=np.int64),
        specs,
        row_feature_columns=row_features,
        candidate_values=candidate_values,
    )
    feature_columns = module.model_feature_columns(row_features)
    assert list(long[feature_columns].columns) == feature_columns
    assert len(long) == 15
    np.testing.assert_allclose(
        frame["candidate_std"].to_numpy(np.float32),
        source[module.CANDIDATE_ORDER].std(axis=1).to_numpy(np.float32),
    )
    assert long["candidate_name"].tolist() == [
        candidate
        for candidate in module.CANDIDATE_ORDER
        for _ in range(3)
    ]
    assert not module.PROTECTED_LABEL_COLUMNS.intersection(long.columns)

    truth = source["last_known_tvt"].to_numpy(np.float32) + 4.0
    errors, within10 = module.candidate_labels(
        truth,
        candidate_values,
        np.asarray([0, 2, 4], dtype=np.int64),
    )
    np.testing.assert_allclose(
        errors.reshape(5, 3)[:, 0],
        np.asarray([3.0, 1.0, 1.0, 3.0, 5.0]),
    )
    assert within10.tolist() == [1] * 15

    leaked = source.assign(target=1.0)
    with pytest.raises(ValueError, match="received protected labels"):
        module.prepare_target_free_wide_features(
            leaked,
            specs,
            configured_row_columns=config["scorer"]["exp111_row_context_columns"],
            configured_global_columns=config["scorer"]["exp111_multiobs_global_columns"],
        )


def test_inner_fold_assignment_is_deterministic_group_safe_and_index_safe(
    module,
) -> None:
    ids = pd.Series(
        [f"id_{value:02d}" for value in range(24)],
        index=np.arange(100, 124),
    )
    wells = pd.Series(
        [f"well_{value // 3:02d}" for value in range(24)],
        index=np.arange(100, 124),
    )
    first, manifest = module.build_inner_fold_assignment(ids, wells)
    second, second_manifest = module.build_inner_fold_assignment(ids, wells)
    np.testing.assert_array_equal(first, second)
    assert manifest == second_manifest
    assert set(first.tolist()) == {0, 1, 2, 3}
    assert len(manifest) == 4
    assert all(row["well_overlap"] == 0 for row in manifest)
    for fold in range(4):
        valid_wells = set(wells.iloc[np.flatnonzero(first == fold)])
        train_wells = set(wells.iloc[np.flatnonzero(first != fold)])
        assert not valid_wells.intersection(train_wells)


def test_stable_sampling_uses_local_sha_seed(module) -> None:
    ids = pd.Series([f"id_{value:03d}" for value in range(40)])
    wells = pd.Series([f"well_{value // 4:03d}" for value in range(40)])
    indices = np.arange(40, dtype=np.int64)
    first, evidence = module.stable_sample_row_indices(
        ids,
        wells,
        indices,
        outer_fold=2,
        inner_fold=1,
        maximum_rows=11,
    )
    second, repeated = module.stable_sample_row_indices(
        ids,
        wells,
        indices[::-1],
        outer_fold=2,
        inner_fold=1,
        maximum_rows=11,
    )
    np.testing.assert_array_equal(first, second)
    assert evidence == repeated
    assert evidence["seed_material"] == (
        "exp396|outer=2|inner=1|candidate_long"
    )
    assert evidence["sampled_rows"] == 11
    assert evidence["global_rng_used"] is False
    other, _ = module.stable_sample_row_indices(
        ids,
        wells,
        indices,
        outer_fold=3,
        inner_fold=1,
        maximum_rows=11,
    )
    assert set(first) != set(other)


def test_medians_are_fit_once_on_inner_train_and_reused(module) -> None:
    columns = [f"feature_{value:02d}" for value in range(48)]
    train = pd.DataFrame(
        np.vstack(
            [
                np.arange(48, dtype=np.float32),
                np.arange(48, dtype=np.float32) + 2.0,
            ]
        ),
        columns=columns,
    )
    train.iloc[0, 0] = np.nan
    medians = module.fit_inner_train_medians(train, columns)
    assert medians.shape == (48,)
    assert medians[0] == pytest.approx(2.0)
    valid = pd.DataFrame(
        np.full((1, 48), 1.0e9, dtype=np.float32),
        columns=columns,
    )
    valid.iloc[0, 0] = np.nan
    transformed = module.apply_fixed_medians(valid, columns, medians)
    assert transformed[0, 0] == pytest.approx(2.0)
    assert np.all(transformed[0, 1:] == np.float32(1.0e9))


def test_fixed_27_derivation_ties_floors_and_roundtrip(module) -> None:
    probability = np.asarray(
        [[0.5, 0.5, 0.2, 0.1, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    predicted_error = np.asarray(
        [[-1.0, 0.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0, 5.0]],
        dtype=np.float32,
    )
    candidate = np.asarray(
        [[100.0, 102.0, 104.0, 106.0, 108.0], [200, 202, 204, 206, 208]],
        dtype=np.float32,
    )
    derived = module.derive_fixed_27_features(
        ids=["id_0", "id_1"],
        wells=["well_0", "well_1"],
        last_known_tvt=np.asarray([90.0, 190.0], dtype=np.float32),
        likpf_mean_d=np.asarray([14.0, 14.0], dtype=np.float32),
        candidate_tvt=candidate,
        probability=probability,
        predicted_error=predicted_error,
    )
    assert list(derived.columns) == ["id", "well", *module.FEATURE_COLUMNS_27]
    assert derived.loc[0, "ll_learned_prob_top1_index"] == 0.0
    assert derived.loc[0, "ll_learned_error_top1_index"] == 0.0
    assert derived.loc[0, "ll_learned_pred_abs_error_pf_ancc"] == 0.0
    expected_weighted = (100.0 * 0.5 + 102.0 * 0.5 + 104.0 * 0.2 + 106.0 * 0.1) / 1.3
    assert derived.loc[
        0,
        "ll_learned_prob_weighted_tvt_minus_last_known_tvt",
    ] == pytest.approx(expected_weighted - 90.0, abs=1.0e-5)
    assert all(dtype == np.dtype("float32") for dtype in derived.iloc[:, 2:].dtypes)

    core = module.score_core_frame(
        ids=["id_0", "id_1"],
        wells=["well_0", "well_1"],
        probability=probability,
        predicted_error=predicted_error,
        downstream_outer_fold=0,
        role="valid",
    )
    roundtrip_probability, roundtrip_error = module.score_core_to_matrices(core)
    np.testing.assert_allclose(roundtrip_probability, probability)
    np.testing.assert_allclose(roundtrip_error, np.maximum(predicted_error, 0.0))


def test_quality_math_and_all_and_gate(module, config) -> None:
    actual_error = np.asarray([[2.0, 15.0], [4.0, 20.0]])
    actual_within10 = (actual_error <= 10.0).astype(float)
    sums = module.score_quality_sums(
        actual_error=actual_error,
        actual_within10=actual_within10,
        predicted_error=actual_error,
        predicted_probability=np.asarray([[0.99, 0.01], [0.99, 0.01]]),
        prior_error=np.full_like(actual_error, 10.0),
        prior_probability=np.full_like(actual_error, 0.5),
    )
    quality_row = module.quality_row_from_sums(sums, outer_fold=0)
    assert quality_row["expected_error_mae_improved"] is True
    assert quality_row["within10_logloss_improved"] is True
    assert quality_row["within10_brier_improved"] is True

    gate_config = copy.deepcopy(config)
    gate_config["validation"]["expected_rows"] = 10
    gate_config["validation"]["expected_wells"] = 5
    partitions = []
    for outer_fold in range(5):
        for role, rows, wells in (("train", 8, 4), ("valid", 2, 1)):
            partitions.append(
                {
                    "downstream_outer_fold": outer_fold,
                    "role": role,
                    "rows": rows,
                    "wells": wells,
                    "duplicate_ids": 0,
                    "score_core_count": 10,
                    "derived_feature_count": 27,
                    "score_core_logical_sha256": "a" * 64,
                    "derived_27_logical_sha256": "b" * 64,
                }
            )
    fold_manifest = [
        {
            "outer_well_overlap": 0,
            "well_overlap": 0,
            "sample_shared_between_objectives": True,
            "global_rng_used": False,
        }
        for _ in range(20)
    ]
    models = [
        {
            "input_feature_count": 48,
            "feature_columns": [f"feature_{value}" for value in range(48)],
            "inner_train_outer_valid_well_overlap": 0,
        }
        for _ in range(40)
    ]
    quality = [
        {
            "outer_fold": fold,
            "expected_error_mae_improved": True,
            "within10_logloss_improved": True,
            "within10_brier_improved": True,
        }
        for fold in range(5)
    ]
    quality.append(
        {
            "outer_fold": "pooled",
            "expected_error_mae_improved": True,
            "within10_logloss_improved": True,
            "within10_brier_improved": True,
        }
    )
    result = {
        "model_manifest": {
            "model_count": 40,
            "median_vector_count": 40,
            "schema_record_count": 40,
            "input_feature_count": 48,
            "models": models,
            "saved_exp111_model_reuse_count": 0,
            "control_retraining_boosters": 0,
        },
        "fold_manifest": pd.DataFrame(fold_manifest),
        "partition_manifest": pd.DataFrame(partitions),
        "quality": pd.DataFrame(quality),
        "source_evidence": {
            "target_free_feature_frame": True,
            "labels_isolated_before_feature_build": True,
        },
        "runtime_seconds": 100.0,
        "peak_rss_gb": 2.0,
    }
    gate = module.evaluate_stage_a_gate(result, config=gate_config)
    assert gate["passed"] is True
    assert gate["stage_b_implementation_authorized"] is False

    result["quality"].loc[
        result["quality"]["outer_fold"].astype(str).eq("pooled"),
        "within10_brier_improved",
    ] = False
    failed = module.evaluate_stage_a_gate(result, config=gate_config)
    assert failed["passed"] is False
    assert failed["score_quality"]["checks"]["within10_brier_pooled_improved"] is False


def test_inference_and_submission_remain_unimplemented(
    inference_module,
    config,
) -> None:
    status = inference_module.validate_inference_is_closed(config)
    current_test = inference_module.current_test_contract(config)
    assert status["boosters_trained"] == 0
    assert status["prediction_generated"] is False
    assert status["submission_generated"] is False
    assert current_test["enabled"] is False
    unauthorized = copy.deepcopy(config)
    unauthorized["execution"]["inference_approved"] = True
    with pytest.raises(RuntimeError, match="not implemented or authorized"):
        inference_module.validate_inference_is_closed(unauthorized)


def test_stage_b_promotion_is_fixed_all_and(module, config, tmp_path) -> None:
    rows = 30
    wells = pd.Series([f"well_{index // 3:02d}" for index in range(rows)])
    folds = np.repeat(np.arange(5, dtype=np.int8), 6)
    actual = np.linspace(100.0, 130.0, rows, dtype=np.float32)
    parent_prediction = actual + np.float32(1.0)
    exp264_prediction = actual + np.float32(1.5)
    new_prediction = actual + np.float32(0.5)
    base = pd.DataFrame(
        {
            "well": wells,
            "md_since": np.resize(
                np.asarray([100.0, 500.0, 1500.0], dtype=np.float32),
                rows,
            ),
        }
    )
    parent = pd.DataFrame(
        {
            "actual_tvt": actual,
            "outer_fold": folds,
            "fold_safe_formation_74_addonly__lgb_mean__pred_tvt": (
                parent_prediction
            ),
        }
    )
    exp264 = pd.DataFrame(
        {
            "actual_tvt": actual,
            "outer_fold": folds,
            "selector_compact_addonly__lgb_mean__pred_tvt": exp264_prediction,
        }
    )
    assignment = pd.DataFrame(
        {
            "well_id": sorted(wells.unique()),
            "verification_like_spatial_role": "valid",
            "verification_like_typewell_purged_role": "valid",
        }
    )
    assignment_path = tmp_path / "hidden.csv"
    assignment.to_csv(assignment_path, index=False)
    guard, fold_metrics, scope_metrics, by_well = (
        module.evaluate_stage_b_promotion(
            config=config,
            base_frame=base,
            parent_oof=parent,
            exp264_oof=exp264,
            new_prediction=new_prediction,
            hidden_assignment_path=assignment_path,
        )
    )
    assert guard["passed"] is True
    assert guard["nonworse_folds_vs_exp287"] == 5
    assert len(fold_metrics) == 5
    assert set(scope_metrics["scope"]) == {
        "near_0_250",
        "mid_250_1000",
        "1000_plus",
        "hidden_like_spatial",
        "hidden_like_typewell_purged",
    }
    assert len(by_well) == 10

    failed, *_ = module.evaluate_stage_b_promotion(
        config=config,
        base_frame=base,
        parent_oof=parent,
        exp264_oof=exp264,
        new_prediction=actual + np.float32(5.0),
        hidden_assignment_path=assignment_path,
    )
    assert failed["passed"] is False
    assert not all(failed["checks"].values())


def test_jupytext_candidates_are_self_contained_and_train_is_canonically_adopted() -> None:
    train_source = TRAIN.read_text()
    inference_source = INFERENCE.read_text()
    assert "__file__" not in train_source
    assert "__file__" not in inference_source
    assert train_source.count("# %% [markdown]") >= 11
    assert "# ## 12. Orchestration" in train_source
    assert "from learned_pf_observation_likelihood_probe import" not in train_source
    assert (EXP_DIR / f"{EXP_NAME}_compact_selfcontained_train.ipynb").is_file()
    assert (EXP_DIR / f"{EXP_NAME}_compact_selfcontained_inference.ipynb").is_file()
    canonical_train = EXP_DIR / f"{EXP_NAME}_train.ipynb"
    candidate_train = EXP_DIR / f"{EXP_NAME}_compact_selfcontained_train.ipynb"
    assert canonical_train.read_bytes() == candidate_train.read_bytes()
    assert (EXP_DIR / f"{EXP_NAME}_inference.ipynb").is_file()
