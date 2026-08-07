from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
EXP = "exp497_strict_public_core_fold_safe_ensemble_on_exp413"
EXP_DIR = ROOT / "experiments" / EXP
SOURCE = EXP_DIR / f"{EXP}_compact_selfcontained_train.py"
INFERENCE_SOURCE = EXP_DIR / f"{EXP}_current_test_inference.py"
SAVED_INFERENCE_SOURCE = EXP_DIR / f"{EXP}_inference.py"
CORE_SOURCE = ROOT / "src" / "strict_public_core.py"


def load_source() -> ModuleType:
    previous = os.environ.get("EXP497_IMPORT_ONLY")
    os.environ["EXP497_IMPORT_ONLY"] = "1"
    try:
        spec = importlib.util.spec_from_file_location("exp497_stage0", SOURCE)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if previous is None:
            os.environ.pop("EXP497_IMPORT_ONLY", None)
        else:
            os.environ["EXP497_IMPORT_ONLY"] = previous


def load_core_source() -> ModuleType:
    spec = importlib.util.spec_from_file_location("exp497_strict_public_core", CORE_SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_spatial_id_guard_reorders_lexical_ids_to_raw_sequence(core: ModuleType) -> None:
    expected = pd.Series(["well_851", "well_1000", "well_1001"], dtype=str, name="id")
    lexical = pd.DataFrame(
        {
            "id": pd.Series(["well_1000", "well_1001", "well_851"], dtype="string"),
            "signal": [1000, 1001, 851],
        }
    )
    aligned = core._align_base_rows_to_expected_ids(lexical, expected, "well")
    assert aligned["id"].tolist() == expected.tolist()
    assert aligned["signal"].tolist() == [851, 1000, 1001]

    with pytest.raises(ValueError, match="id set mismatch"):
        core._align_base_rows_to_expected_ids(lexical.iloc[:-1], expected, "well")


@pytest.fixture(scope="module")
def stage0() -> ModuleType:
    return load_source()


@pytest.fixture(scope="module")
def core() -> ModuleType:
    return load_core_source()


@pytest.fixture(scope="module")
def config() -> dict:
    value = yaml.safe_load((EXP_DIR / "config.yaml").read_text())
    assert isinstance(value, dict)
    return value


@pytest.fixture(scope="module")
def inventory() -> dict:
    value = yaml.safe_load((EXP_DIR / "public_source_inventory.yaml").read_text())
    assert isinstance(value, dict)
    return value


def test_stage0_static_contract_freezes_cost_and_feature_counts(
    stage0: ModuleType,
) -> None:
    observed = stage0.STATIC_CONTRACT
    assert observed["status"] == "pass"
    assert observed["model_cost"] == {
        "scientific_variants": 1,
        "ml_branches": 2,
        "configs_per_branch": 5,
        "lightgbm_configs_per_branch": 3,
        "catboost_configs_per_branch": 2,
        "outer_folds": 5,
        "inner_folds": 4,
        "planned_lightgbm_boosters": 120,
        "planned_catboost_boosters": 80,
        "planned_total_boosters": 200,
        "planned_ridge_models": 10,
        "exp413_parent_retraining": 0,
        "exp413_selector_retraining": 0,
        "exp413_signed_selector_retraining": 0,
        "exp413_tvt_model_retraining": 0,
    }
    assert observed["feature_schema"]["sp45_feature_count"] == 195
    assert observed["feature_schema"]["learned_total_feature_count"] == 205
    assert observed["feature_schema"]["both_surfaces_kept_in_ram"] is False


def test_physical_execution_inventory_is_exact(stage0: ModuleType) -> None:
    observed = stage0.STATIC_CONTRACT["physical_execution"]
    assert observed["selector_likelihood_pf_seed_banks"] == 773
    assert observed["learned_likelihood_pf_seed_banks"] == 773
    assert observed["likelihood_pf_total_seed_well_runs"] == 197_888
    assert observed["likelihood_pf_total_particle_starts"] == 98_944_000
    assert observed["selector_beam_well_runs"] == 10_822
    assert observed["learned_beam_well_runs"] == 5_411
    assert observed["total_beam_well_runs"] == 16_233
    assert observed["ncc_well_window_runs"] == 2_319
    assert observed["formation_plane_pool_fits"] == 5
    assert observed["dense_ancc_pool_fits"] == 5
    assert observed["fold_surface_well_queries_per_pool_family"] == 3_865


def test_implementation_time_source_audit_matches_inventory(
    config: dict,
    inventory: dict,
) -> None:
    audit = json.loads((EXP_DIR / "public_source_audit.json").read_text())
    source = inventory["source"]
    assert audit["source"]["converted_py_sha256"] == source["converted_source_sha256"]
    assert audit["source"]["converted_py_bytes"] == source["converted_source_bytes"]
    assert audit["source"]["converted_py_lines"] == source["converted_source_lines"]
    assert (
        config["data"]["public_source_py"]["expected_sha256"] == source["converted_source_sha256"]
    )
    assert audit["route_code_extracted"] is True
    assert audit["kaggle_run_performed"] is False


def test_reference_source_scan_is_sha_and_symbol_fail_closed(
    stage0: ModuleType,
    inventory: dict,
    tmp_path: Path,
) -> None:
    source_text = "def kept_symbol():\n    return 1\n"
    source_path = tmp_path / "reference.py"
    source_path.write_text(source_text)
    synthetic = copy.deepcopy(inventory)
    synthetic["source"]["converted_source_sha256"] = hashlib.sha256(
        source_path.read_bytes()
    ).hexdigest()
    synthetic["source"]["converted_source_bytes"] = len(source_path.read_bytes())
    synthetic["source"]["converted_source_lines"] = len(source_text.splitlines())
    synthetic["required_symbols"] = [
        {"name": "kept_symbol", "kind": "function", "definition_lines": [1]}
    ]
    audit = stage0.scan_reference_source(source_path, synthetic)
    assert audit["status"] == "pass"
    assert audit["required_symbols"][0]["definition_lines"] == [1]

    source_path.write_text(source_text + "# mutation\n")
    with pytest.raises(ValueError, match="SHA mismatch"):
        stage0.scan_reference_source(source_path, synthetic)


def test_decontamination_scan_rejects_forbidden_route_symbol(
    stage0: ModuleType,
    inventory: dict,
) -> None:
    assert stage0.scan_decontaminated_executable(
        "def allowed_route():\n    return 1\n", inventory
    ) == {label: 0 for label in inventory["decontaminated_executable_forbidden_patterns"]}
    with pytest.raises(ValueError, match="Decontamination scan failed"):
        stage0.scan_decontaminated_executable(
            "def route():\n    return visible_prefix_value\n", inventory
        )


def test_stable_seed_depends_only_on_immutable_key(stage0: ModuleType) -> None:
    kwargs = {
        "experiment": EXP,
        "stage": "P",
        "split": "oof",
        "outer_fold": 2,
        "inner_fold": 1,
        "family": "likelihood_pf",
        "well_id": "well_a",
        "seed_index": 7,
        "base_seed": 42,
    }
    first = stage0.stable_seed(**kwargs)
    second = stage0.stable_seed(**kwargs)
    assert first == second
    assert 0 <= first < 2**32 - 1
    changed = dict(kwargs)
    changed["well_id"] = "well_b"
    assert stage0.stable_seed(**changed) != first
    assert len({stage0.stable_seed(**(kwargs | {"seed_index": i})) for i in range(128)}) == 128


def synthetic_parent_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for fold in range(5):
        for well_index in range(2):
            well = f"w{fold}{well_index}"
            for row_idx in range(3 + fold + well_index):
                rows.append(
                    {
                        "well_id": well,
                        "row_idx": row_idx,
                        "fold": fold,
                        "actual_tvt": 100.0 + fold + row_idx,
                        "exp413_pred_tvt": 100.1 + fold + row_idx,
                    }
                )
    return pd.DataFrame(rows)


def test_parent_oof_canonicalization_and_fold_contract(stage0: ModuleType) -> None:
    raw = pd.DataFrame(
        {
            "id": ["aa_0", "aa_1", "bb_0", "bb_1"],
            "well": ["aa", "aa", "bb", "bb"],
            "outer_fold": [0, 0, 1, 1],
            "actual_tvt": [1.0, 2.0, 3.0, 4.0],
            "frozen_prediction": [1.1, 2.1, 3.1, 4.1],
        }
    )
    canonical = stage0.canonicalize_parent_oof(raw, "frozen_prediction")
    assert canonical.columns.tolist() == [
        "well_id",
        "row_idx",
        "fold",
        "actual_tvt",
        "exp413_pred_tvt",
    ]
    assert canonical["row_idx"].tolist() == [0, 1, 0, 1]
    with pytest.raises(ValueError, match="0..4"):
        stage0.validate_parent_fold_frame(
            canonical,
            expected_rows=4,
            expected_wells=2,
            expected_folds=5,
        )


def test_nested_inner_folds_and_spatial_pools_exclude_outer_valid(
    stage0: ModuleType,
) -> None:
    parent = synthetic_parent_frame()
    outer_audit = stage0.validate_parent_fold_frame(
        parent,
        expected_rows=len(parent),
        expected_wells=10,
        expected_folds=5,
    )
    assert outer_audit["wells"] == 10
    manifest = stage0.build_inner_fold_manifest(parent, outer_folds=5, inner_folds=4)
    audit = stage0.validate_inner_fold_manifest(
        parent,
        manifest,
        outer_folds=5,
        inner_folds=4,
    )
    assert audit["status"] == "pass"
    assert len(manifest) == 5 * 8
    assert all(row["outer_valid_overlap"] == 0 for row in audit["outer_folds"])

    ledger = stage0.build_spatial_pool_ledger(parent, outer_folds=5)
    assert ledger["pool_outer_valid_overlap"].eq(0).all()
    assert ledger["pool_wells"].eq(8).all()
    assert ledger["outer_valid_wells"].eq(2).all()
    assert ledger["query_wells"].eq(10).all()
    assert ledger["self_well_exclusion_required"].all()


def test_well_shape_selector_fits_only_supplied_outer_train(
    stage0: ModuleType,
    config: dict,
) -> None:
    variants = config["public_core"]["physical_selector"]["candidate_variants"]
    metadata = pd.DataFrame(
        {
            "well_id": [f"w{i:02d}" for i in range(24)],
            "n_eval": np.arange(24) * 100 + 100,
            "z_span": np.linspace(50.0, 250.0, 24),
        }
    )
    losses: list[dict[str, object]] = []
    for well_index, well in enumerate(metadata["well_id"]):
        winner = variants[well_index % len(variants)]
        for variant_index, variant in enumerate(variants):
            losses.append(
                {
                    "well_id": well,
                    "variant": variant,
                    "rows": 100,
                    "sse": float(variant_index + (0 if variant == winner else 100)),
                }
            )
    selector = stage0.fit_well_shape_selector(
        metadata,
        pd.DataFrame(losses),
        variant_order=variants,
        minimum_bin_wells=2,
    )
    valid_metadata = pd.DataFrame(
        {
            "well_id": ["valid_a", "valid_b"],
            "n_eval": [50, 10_000],
            "z_span": [40.0, 300.0],
            "unread_truth": [-1.0e9, 1.0e9],
        }
    )
    selected = stage0.apply_well_shape_selector(valid_metadata, selector)
    assert selected["selected_variant"].isin(variants).all()
    changed_truth = valid_metadata.copy()
    changed_truth["unread_truth"] *= -1
    pd.testing.assert_frame_equal(
        selected,
        stage0.apply_well_shape_selector(changed_truth, selector),
    )

    incomplete = pd.DataFrame(losses).iloc[:-1].copy()
    with pytest.raises(ValueError, match="every variant for every well"):
        stage0.fit_well_shape_selector(
            metadata,
            incomplete,
            variant_order=variants,
            minimum_bin_wells=2,
        )


def test_truth_attach_requires_unchanged_frozen_prediction(stage0: ModuleType) -> None:
    prediction = pd.DataFrame(
        {
            "well_id": ["a", "a", "b"],
            "row_idx": [0, 1, 0],
            "fold": [0, 0, 1],
            "strict_core_pred": [10.0, 11.0, 12.0],
        }
    )
    manifest = stage0.freeze_target_free_prediction(
        prediction,
        keys=["well_id", "row_idx", "fold"],
        prediction_columns=["strict_core_pred"],
    )
    truth = pd.DataFrame(
        {
            "well_id": ["a", "a", "b"],
            "row_idx": [0, 1, 0],
            "fold": [0, 0, 1],
            "actual_tvt": [10.5, 10.5, 12.5],
        }
    )
    attached = stage0.attach_truth_after_freeze(
        prediction,
        truth,
        manifest,
        truth_columns=["actual_tvt"],
    )
    assert attached["actual_tvt"].tolist() == [10.5, 10.5, 12.5]

    mutated = prediction.copy()
    mutated.loc[0, "strict_core_pred"] += 0.01
    with pytest.raises(ValueError, match="changed before truth attach"):
        stage0.attach_truth_after_freeze(
            mutated,
            truth,
            manifest,
            truth_columns=["actual_tvt"],
        )

    contaminated = prediction.assign(actual_tvt=[10.5, 10.5, 12.5])
    with pytest.raises(ValueError, match="after truth attach"):
        stage0.freeze_target_free_prediction(
            contaminated,
            keys=["well_id", "row_idx", "fold"],
            prediction_columns=["strict_core_pred"],
        )


def test_meta_blend_weight_is_fitted_on_other_four_folds(stage0: ModuleType) -> None:
    rng = np.random.default_rng(42)
    rows = 500
    folds = np.arange(rows) % 5
    base = rng.normal(100.0, 4.0, rows)
    auxiliary = base + rng.normal(0.0, 2.0, rows)
    truth = 0.8 * base + 0.2 * auxiliary
    frame = pd.DataFrame(
        {
            "fold": folds,
            "actual_tvt": truth,
            "exp413": base,
            "strict_core": auxiliary,
        }
    )
    result = stage0.crossfit_constant_blend(
        frame,
        truth_column="actual_tvt",
        base_column="exp413",
        auxiliary_column="strict_core",
        fold_column="fold",
        folds=5,
        lower=0.0,
        upper=0.30,
    )
    np.testing.assert_allclose(result["meta_fold_weights"], 0.2, atol=1.0e-12)
    np.testing.assert_allclose(result["crossfit_prediction"], truth, atol=1.0e-12)
    assert result["deployment_weight"] == pytest.approx(0.2)
    assert result["full_oof_refit"] is False

    mutated = frame.copy()
    mutated.loc[mutated["fold"].eq(0), "actual_tvt"] += 10_000.0
    mutated_result = stage0.crossfit_constant_blend(
        mutated,
        truth_column="actual_tvt",
        base_column="exp413",
        auxiliary_column="strict_core",
        fold_column="fold",
        folds=5,
        lower=0.0,
        upper=0.30,
    )
    assert mutated_result["meta_fold_weights"][0] == pytest.approx(result["meta_fold_weights"][0])
    np.testing.assert_array_equal(
        mutated_result["crossfit_prediction"][folds == 0],
        result["crossfit_prediction"][folds == 0],
    )


def test_parent_sha_contract_is_fully_frozen(config: dict) -> None:
    parent = config["data"]["parent_exp413"]
    assert parent["expected_final_oof_sha256"] == (
        "9bd2d17778b3b27d771b12cbff72def8b87e6cdf14062e1c0ba192434cef4a9d"
    )
    assert parent["expected_fold_metrics_sha256"] == (
        "82e70b6674f218f2892d6e5f70e327dfcbbdaf0fa5e431c4e07231009e9e2d8f"
    )
    assert parent["expected_scope_metrics_sha256"] == (
        "c89add97cd4cae628b79774615a717e4cfbffe7b65a4a68c58b2c2e2737948ed"
    )
    assert parent["expected_hidden_like_metrics_sha256"] == (
        "eafa3546e4ea5c0d180d380f7fe2c39b5cac970ea4c8097b68b077017da1f1b8"
    )
    assert parent["expected_by_well_sha256"] == (
        "e82c6908ed2caa9b3e5c1664bc66a3226b3bc6d9284f4863bd4fa941ae32d080"
    )
    assert parent["control_retraining_allowed"] is False


def test_run_gate_records_explicit_execution_approval(
    stage0: ModuleType,
    config: dict,
) -> None:
    assert config["implementation"]["approved"] is True
    assert config["stage0"]["kaggle_run_approved"] is True
    assert config["implementation"]["kaggle_run_approved"] is True
    assert config["implementation"]["canonical_train_notebook_adopted"] is True
    assert config["implementation"]["model_serialization_rerun_approved"] is True
    assert (
        config["implementation"]["model_serialization_approval_source"]
        == "user_message_save_model_weights_and_rerun_2026_08_04"
    )
    assert "user_message_execute_exp497_2026_08_01" in config["implementation"]["approval_source"]
    assert (
        "user_message_use_kaggle_gpu_not_colab_2026_08_01"
        in config["implementation"]["approval_source"]
    )
    stage0.require_stage0_run_authorization(config)


def test_route_training_inventory_is_exact(core: ModuleType, config: dict) -> None:
    observed = core.training_inventory()
    assert observed == {
        "scientific_variants": 1,
        "branches": 2,
        "outer_folds": 5,
        "inner_folds": 4,
        "lightgbm_configs_per_branch": 3,
        "catboost_configs_per_branch": 2,
        "planned_lightgbm_boosters": 120,
        "planned_catboost_boosters": 80,
        "planned_total_boosters": 200,
        "planned_ridge_models": 10,
        "exp413_retraining": 0,
    }
    assert config["stages"]["stage_m"]["accelerator"] == "kaggle_gpu"
    assert config["stages"]["stage_m"]["boosters_per_shard"] == 40


def test_stage_i_prediction_only_inventory_and_fixed_weight(config: dict) -> None:
    stage_i = config["stages"]["stage_i"]
    assert stage_i == {
        "implemented": True,
        "notebook_kind": "current_test_inference",
        "accelerator": "kaggle_gpu",
        "mode": "prediction_only_diagnostic_override",
        "lightgbm_boosters": 24,
        "catboost_boosters": 16,
        "total_boosters": 40,
        "ridge_models": 2,
        "serialize_booster_models": True,
        "serialized_lightgbm_models": 24,
        "serialized_catboost_models": 16,
        "serialize_ridge_weights": True,
        "serialization_reload_parity_max_abs": 1.0e-5,
        "exp413_retraining": 0,
        "exp413_reinference": 0,
        "submission_enabled": False,
    }
    inference = config["inference"]
    weights = np.asarray(inference["meta_fold_weights"], dtype=np.float64)
    assert len(weights) == 5
    assert inference["deployment_weight"] == pytest.approx(float(np.median(weights)))
    assert inference["selected_train_anchor_unchanged"] == "exp413"
    assert inference["generate_submission_csv"] is False
    assert inference["external_submission"] is False


def test_saved_model_inference_inventory_and_authorization(config: dict) -> None:
    stage = config["stages"]["stage_i_saved_model_inference"]
    deployment = config["inference"]["saved_model_deployment"]
    assert stage == {
        "implemented": True,
        "notebook_kind": "inference",
        "canonical_notebook_adopted": True,
        "accelerator": "kaggle_gpu",
        "fitted_boosters": 0,
        "loaded_exp497_boosters": 40,
        "loaded_exp497_ridge_models": 2,
        "loaded_exp413_boosters": 75,
        "exp413_retraining": 0,
        "submission_enabled": True,
        "external_submission_enabled": False,
        "status": "complete_validated",
        "kaggle_kernel_version": 2,
        "kaggle_initial_status": "RUNNING",
        "kaggle_final_status": "COMPLETE",
        "rows": 14151,
        "wells": 3,
        "saved_model_inference_elapsed_seconds": 33.509,
        "visible_parity_passed": True,
        "submission_sha256": (
            "04ca2e2f80f45bced1e22bd68a58002b4cb7c7e5b19510932375cdccafa6680a"
        ),
    }
    assert deployment["approved"] is True
    assert deployment["exp497_booster_training_count"] == 0
    assert deployment["exp413_booster_training_count"] == 0
    assert deployment["public_test_exp413_sidecar_allowed"] is False
    assert deployment["generate_submission_csv"] is True
    assert deployment["external_submission"] is False
    assert deployment["visible_strict_public_core_max_abs"] == pytest.approx(0.002)
    assert deployment["visible_blend_max_abs"] == pytest.approx(0.02)
    assert deployment["isolate_exp413_intermediate_submission"] is True
    assert deployment["serialized_model_set_sha256"] == (
        "dcc2166f4bd5731364efe0b3fb848a46cf87f8133cbe78890658a1062c604626"
    )
    assert set(deployment["artifact_sha256"]) == set(core_artifact_files())


def core_artifact_files() -> dict[str, str]:
    return {
        "model_manifest": "stage_i_full_fit_model_manifest.json",
        "ridge_weights": "stage_i_ridge_weights.json",
        "weights": "stage_i_weights.json",
        "selector_policy": "stage_i_selector_policy.json",
        "feature_schema": "stage_i_feature_schema.csv",
        "reproducibility_manifest": "stage_i_reproducibility_manifest.json",
    }


def test_saved_model_visible_parity_uses_component_specific_tolerances(
    core: ModuleType,
) -> None:
    observed = core.validate_stage_i_visible_parity(
        strict_public_core_max_abs=0.0012812500008294592,
        blend_max_abs=0.01419531250030559,
        strict_public_core_tolerance=0.002,
        blend_tolerance=0.02,
        exp413_max_abs=0.016,
    )
    assert observed["passed"] is True
    with pytest.raises(ValueError, match="visible saved-model parity failed"):
        core.validate_stage_i_visible_parity(
            strict_public_core_max_abs=0.0021,
            blend_max_abs=0.014,
            strict_public_core_tolerance=0.002,
            blend_tolerance=0.02,
        )
    with pytest.raises(ValueError, match="visible saved-model parity failed"):
        core.validate_stage_i_visible_parity(
            strict_public_core_max_abs=0.001,
            blend_max_abs=0.0201,
            strict_public_core_tolerance=0.002,
            blend_tolerance=0.02,
        )


def test_stage_i_notebook_is_kaggle_prediction_only_and_dynamic() -> None:
    source_text = INFERENCE_SOURCE.read_text()
    assert "run_stage_i_current_test(" in source_text
    assert "build_stage_i_test_feature_frame(" in source_text
    assert 'f"stage_p_fold{fold}_physical_features.parquet"' in source_text
    assert 'f"stage_p_fold{fold}_features.parquet"' not in source_text
    assert 'summary["fitted_boosters"] != 40' in source_text
    assert 'summary["serialized_model_count"] != 40' in source_text
    assert 'summary["serialized_lightgbm_count"] != 24' in source_text
    assert 'summary["serialized_catboost_count"] != 16' in source_text
    assert 'not (OUTPUT_DIR / row["model_file"]).is_file()' in source_text
    assert 'summary["exp413_retraining"] != 0' in source_text
    assert 'summary["exp413_reinference"] != 0' in source_text
    assert 'promotion_gate["passed"] is not False' in source_text
    assert "kaggle competitions submit" not in source_text.lower()
    assert "to_csv(\"submission.csv\"" not in source_text
    assert "/content" not in source_text
    assert "colab" not in source_text.lower()
    assert "expected_test_rows" not in source_text
    assert "expected_test_wells" not in source_text


def test_route_feature_schema_is_exact(core: ModuleType) -> None:
    base_columns = [f"base_{index:03d}" for index in range(195)]
    frame = pd.DataFrame(
        columns=[
            "id",
            "well",
            "target",
            "outer_fold",
            "actual_tvt",
            *base_columns,
            *core.LEARNED_LIKPF_COLUMNS,
            "selector_beam_mean",
            *(f"selector__{variant}" for variant in core.SELECTOR_VARIANTS),
        ]
    )
    base, learned = core.base_and_learned_feature_columns(frame)
    assert base == base_columns
    assert learned == [*base_columns, *core.LEARNED_LIKPF_COLUMNS]


def test_selector_policy_does_not_read_nonfit_well_truth(core: ModuleType) -> None:
    train_wells = {f"train_{index:02d}" for index in range(12)}
    held_wells = {"held_a", "held_b"}
    all_wells = sorted(train_wells | held_wells)
    metadata = pd.DataFrame(
        {
            "well": all_wells,
            "n_eval": np.arange(len(all_wells)) + 100,
            "z_span": np.linspace(50.0, 250.0, len(all_wells)),
        }
    )
    rows = []
    for well in all_wells:
        for row_index in range(3):
            row = {
                "well": well,
                "last_known_tvt": 100.0,
                "target": float(row_index),
            }
            for variant_index, variant in enumerate(core.SELECTOR_VARIANTS):
                row[f"selector__{variant}"] = 100.0 + row_index + 0.1 * variant_index
            rows.append(row)
    candidate_rows = pd.DataFrame(rows)
    first = core.fit_selector_policy(
        metadata,
        candidate_rows,
        train_wells,
        minimum_bin_wells=2,
    )
    mutated = candidate_rows.copy()
    mutated.loc[mutated["well"].isin(held_wells), "target"] += 1_000_000.0
    second = core.fit_selector_policy(
        metadata,
        mutated,
        train_wells,
        minimum_bin_wells=2,
    )
    assert first == second


def test_current_test_selector_uses_test_metadata_when_well_ids_overlap(
    core: ModuleType,
) -> None:
    variants = list(core.SELECTOR_VARIANTS)
    frame = pd.DataFrame({"well": ["shared", "shared"]})
    for index, variant in enumerate(variants):
        frame[f"selector__{variant}"] = np.float32(100.0 + index)
    train_metadata = pd.DataFrame(
        {"well": ["shared"], "n_eval": [1_000], "z_span": [300.0]}
    )
    test_metadata = pd.DataFrame(
        {"well": ["shared"], "n_eval": [10], "z_span": [50.0]}
    )
    policy = {
        "n_eval_threshold": 100.0,
        "z_span_thresholds": [100.0, 200.0],
        "global_variant": variants[0],
        "mapping": {str(code): variants[code % len(variants)] for code in range(6)},
    }

    prediction = core.apply_selector_policy(frame, test_metadata, policy)
    np.testing.assert_array_equal(prediction, np.full(2, 100.0, dtype=np.float32))

    combined = pd.concat([train_metadata, test_metadata], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate wells"):
        core.apply_selector_policy(frame, combined, policy)


def test_stage_i_passes_only_current_test_metadata_to_selector() -> None:
    source_text = CORE_SOURCE.read_text()
    assert (
        "apply_selector_policy(test_frame, test_well_metadata, deployment_policy)"
        in source_text
    )
    assert "combined_metadata = pd.concat" not in source_text


def test_stage_i_serialized_model_manifest_contract(
    core: ModuleType,
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "stage_i_models"
    model_dir.mkdir()
    rows = []
    for index in range(40):
        kind = "lightgbm" if index < 24 else "catboost"
        suffix = ".txt" if kind == "lightgbm" else ".cbm"
        relative = Path("stage_i_models") / f"model_{index:02d}{suffix}"
        path = tmp_path / relative
        path.write_bytes(f"serialized-model-{index}".encode())
        rows.append(
            {
                "kind": kind,
                "model_file": relative.as_posix(),
                "model_sha256": core.sha256_file(path),
                "model_bytes": path.stat().st_size,
                "serialization_max_abs": 0.0,
            }
        )

    observed = core.validate_stage_i_serialized_model_manifest(tmp_path, rows)
    assert observed["serialized_model_count"] == 40
    assert observed["serialized_lightgbm_count"] == 24
    assert observed["serialized_catboost_count"] == 16
    assert observed["serialized_model_bytes"] > 0
    assert len(observed["serialized_model_set_sha256"]) == 64

    rows[0]["model_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA mismatch"):
        core.validate_stage_i_serialized_model_manifest(tmp_path, rows)


def test_stage_i_ridge_stack_requires_five_ordered_configs(core: ModuleType) -> None:
    matrix = np.arange(15, dtype=np.float32).reshape(3, 5)
    ridge = {"coef": [0.1, 0.2, 0.3, 0.4, 0.5], "intercept": -0.25}
    observed = core.apply_stage_i_ridge_stack(matrix, ridge)
    expected = matrix @ np.asarray(ridge["coef"], dtype=np.float32) - np.float32(0.25)
    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=1e-6)
    with pytest.raises(ValueError, match="five config columns"):
        core.apply_stage_i_ridge_stack(matrix[:, :4], ridge)


def test_stage_i_saved_artifact_loader_is_sha_and_schema_fail_closed(
    core: ModuleType,
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "stage_i_models"
    model_dir.mkdir()
    branch_rows: dict[str, list[dict[str, object]]] = {
        "sp45_residual": [],
        "learned_trajectory": [],
    }
    ordered_rows: list[dict[str, object]] = []
    for branch in branch_rows:
        for config_name in core.STAGE_I_STACK_CONFIGS:
            kind = "lightgbm" if config_name.startswith("lgb") else "catboost"
            suffix = ".txt" if kind == "lightgbm" else ".cbm"
            for inner in range(4):
                relative = Path("stage_i_models") / f"{branch}__{config_name}__inner{inner}{suffix}"
                path = tmp_path / relative
                path.write_bytes(f"{branch}-{config_name}-{inner}".encode())
                row = {
                    "branch": branch,
                    "config": config_name,
                    "kind": kind,
                    "inner_fold": inner,
                    "best_iteration": 1,
                    "model_file": relative.as_posix(),
                    "model_sha256": core.sha256_file(path),
                    "model_bytes": path.stat().st_size,
                    "serialization_max_abs": 0.0,
                }
                branch_rows[branch].append(row)
                ordered_rows.append(row)
    serialization = core.validate_stage_i_serialized_model_manifest(tmp_path, ordered_rows)
    ridge = {
        "coef": [0.1, 0.2, 0.3, 0.2, 0.2],
        "intercept": 0.0,
        "alpha": 1.66,
        "positive": True,
    }
    manifest = {
        "sp45_models": branch_rows["sp45_residual"],
        "learned_models": branch_rows["learned_trajectory"],
        "sp45_ridge": ridge,
        "learned_ridge": ridge,
        "fitted_boosters": 40,
        "fitted_ridge_models": 2,
        "base_features": ["base_feature"],
        "learned_features": ["base_feature", "likpf_scale_5"],
        **serialization,
    }
    ridge_weights = {
        "sp45_residual": ridge,
        "learned_trajectory": ridge,
        "ridge_model_count": 2,
    }
    weights = {
        "sp45_model_weight": 0.45,
        "projection_weight": 0.5,
        "learned_model_weight": 0.8,
        "projected_sp45_weight": 0.5,
        "meta_fold_weights": [0.1, 0.12, 0.14, 0.16, 0.18],
        "deployment_weight": 0.14,
    }
    selector = [
        {
            "role": "current_test",
            "inner_fold": None,
            "n_eval_threshold": 100.0,
            "z_span_thresholds": [100.0, 200.0],
            "global_variant": core.SELECTOR_VARIANTS[0],
            "mapping": {str(index): core.SELECTOR_VARIANTS[0] for index in range(6)},
        }
    ]
    core.write_json(tmp_path / core.STAGE_I_ARTIFACT_FILES["model_manifest"], manifest)
    core.write_json(tmp_path / core.STAGE_I_ARTIFACT_FILES["ridge_weights"], ridge_weights)
    core.write_json(tmp_path / core.STAGE_I_ARTIFACT_FILES["weights"], weights)
    core.write_json(tmp_path / core.STAGE_I_ARTIFACT_FILES["selector_policy"], selector)
    pd.DataFrame(
        [
            {"branch": "sp45_residual", "feature_index": 0, "feature": "base_feature"},
            {"branch": "learned_trajectory", "feature_index": 0, "feature": "base_feature"},
            {"branch": "learned_trajectory", "feature_index": 1, "feature": "likpf_scale_5"},
        ]
    ).to_csv(tmp_path / core.STAGE_I_ARTIFACT_FILES["feature_schema"], index=False)
    core.write_json(
        tmp_path / core.STAGE_I_ARTIFACT_FILES["reproducibility_manifest"],
        {"serialized_model_set_sha256": serialization["serialized_model_set_sha256"]},
    )
    expected_sha = {
        name: core.sha256_file(tmp_path / filename)
        for name, filename in core.STAGE_I_ARTIFACT_FILES.items()
    }
    observed = core.load_stage_i_saved_inference_artifacts(
        tmp_path,
        expected_sha256=expected_sha,
        expected_model_set_sha256=serialization["serialized_model_set_sha256"],
    )
    assert observed["serialization"]["serialized_model_count"] == 40
    assert observed["feature_lists"]["learned_trajectory"] == [
        "base_feature",
        "likpf_scale_5",
    ]

    expected_sha["weights"] = "0" * 64
    with pytest.raises(ValueError, match="artifact SHA mismatch"):
        core.load_stage_i_saved_inference_artifacts(
            tmp_path,
            expected_sha256=expected_sha,
            expected_model_set_sha256=serialization["serialized_model_set_sha256"],
        )


def test_stage_i_saves_and_reloads_all_booster_families() -> None:
    source_text = CORE_SOURCE.read_text()
    assert "model.booster_.save_model(" in source_text
    assert 'model.save_model(str(model_path), format="cbm")' in source_text
    assert "Booster(model_file=str(model_path))" in source_text
    assert 'reloaded_model.load_model(str(model_path), format="cbm")' in source_text
    assert 'model_output_dir = output_dir / "stage_i_models"' in source_text
    assert 'ridge_weights_path = output_dir / "stage_i_ridge_weights.json"' in source_text


def test_saved_model_inference_candidate_is_dynamic_and_training_free() -> None:
    source_text = SAVED_INFERENCE_SOURCE.read_text()
    assert "generate_dynamic_exp413_prediction()" in source_text
    assert "build_stage_i_test_feature_frame(" in source_text
    assert "run_stage_i_saved_model_inference(" in source_text
    assert 'submission_output_path=KAGGLE_WORKING / "submission.csv"' in source_text
    assert "isolate_exp413_intermediate_submission(" in source_text
    assert 'OUTPUT_DIR / "exp413_intermediate_submission.csv"' in source_text
    assert "source_path.replace(destination_path)" in source_text
    assert 'deployment["visible_strict_public_core_max_abs"]' in source_text
    assert 'deployment["visible_blend_max_abs"]' in source_text
    assert '"exp497_fitted_boosters": 0' in source_text
    assert '"exp413_fitted_boosters": 0' in source_text
    assert "exp413_current_test_predictions.csv.gz" not in source_text
    assert "kaggle competitions submit" not in source_text.lower()
    assert "/content" not in source_text
    assert "colab" not in source_text.lower()
    assert "expected_test_rows" not in source_text
    assert "expected_test_wells" not in source_text
    assert "__file__" not in source_text


def test_kaggle_shard_sources_are_complete_and_no_colab_path() -> None:
    kinds = [
        "pfbeam_features",
        *(f"pfbeam_features_fold{fold}" for fold in range(5)),
        *(f"train_fold{fold}" for fold in range(5)),
        "train_aggregate",
        "current_test_inference",
    ]
    combined = ""
    for kind in kinds:
        script = EXP_DIR / f"{EXP}_{kind}.py"
        notebook = EXP_DIR / f"{EXP}_{kind}.ipynb"
        assert script.is_file() and notebook.is_file()
        text = script.read_text()
        assert "__file__" not in text
        combined += text
    assert "run_stage_p_shard" in combined
    assert combined.count("SHARD_FOLD = ") == 5
    assert combined.count("run_stage_m_outer(") == 5
    assert "run_stage_e" in combined
    assert "run_stage_i_current_test" in combined
    assert "colab" not in combined.lower()


def test_compact_candidate_is_not_file_path_dependent() -> None:
    source_text = SOURCE.read_text()
    assert "__file__" not in source_text
    assert "from settings import" not in source_text
    assert "import settings" not in source_text
