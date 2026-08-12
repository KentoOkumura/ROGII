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


ROOT = Path(__file__).resolve().parents[3]
EXP_NAME = "exp402_fold_safe_grwr_5_addonly_on_exp287"
EXP_DIR = ROOT / "experiments" / EXP_NAME
TRAIN = EXP_DIR / f"{EXP_NAME}_compact_selfcontained_train.py"
STAGE1_TRAIN = (
    EXP_DIR / f"{EXP_NAME}_compact_selfcontained_stage1_train.py"
)
INFERENCE = EXP_DIR / f"{EXP_NAME}_compact_selfcontained_inference.py"
AGGREGATE = EXP_DIR / f"{EXP_NAME}_train_aggregate.py"
CONFIG = EXP_DIR / "config.yaml"
EXP218_DIR = (
    ROOT
    / "experiments"
    / "exp218_gr_wavelet_rotation_confidence_features_on_exp148"
)
EXP218_SOURCE = EXP218_DIR / "gr_wavelet_rotation_confidence_features_on_exp148.py"
EXP218_CONFIG = EXP218_DIR / "config.yaml"


def load_module(name: str, path: Path, env_name: str | None = None):
    previous = os.environ.get(env_name) if env_name else None
    if env_name:
        os.environ[env_name] = "1"
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        if env_name:
            if previous is None:
                os.environ.pop(env_name, None)
            else:
                os.environ[env_name] = previous


@pytest.fixture(scope="module")
def module():
    return load_module("exp402_train_test", TRAIN, "EXP402_IMPORT_ONLY")


@pytest.fixture(scope="module")
def stage1_module():
    return load_module(
        "exp402_stage1_train_test",
        STAGE1_TRAIN,
        "EXP402_STAGE1_IMPORT_ONLY",
    )


@pytest.fixture(scope="module")
def inference_module():
    return load_module("exp402_inference_test", INFERENCE)


@pytest.fixture(scope="module")
def exp218_module():
    return load_module("exp218_formula_reference_test", EXP218_SOURCE)


@pytest.fixture()
def config() -> dict:
    value = yaml.safe_load(CONFIG.read_text())
    assert isinstance(value, dict)
    return value


def historical_stage0_config(config: dict) -> dict:
    value = copy.deepcopy(config)
    value["implementation"]["training_implemented"] = False
    value["implementation"]["stage_1_canonical_notebook_adopted"] = False
    value["model"]["training"]["enabled"] = False
    value["execution"]["current_stage"] = "zero_booster_preflight"
    value["execution"]["run_preflight"] = True
    value["execution"]["training_implementation_approved"] = False
    value["execution"]["training_run_approved"] = False
    value["execution"]["kaggle_push_approved"] = False
    value["execution"]["run_train"] = False
    value["runtime"]["device"] = "cpu"
    value["runtime"]["kaggle"]["enable_gpu"] = False
    value["runtime"]["kaggle"]["machine_shape"] = ""
    return value


def make_grwr_inputs(module, rows: int = 6):
    index = np.arange(rows, dtype=np.float32)
    ids = [f"well_{value // 3}_{value:03d}" for value in range(rows)]
    wells = [f"well_{value // 3}" for value in range(rows)]
    last = np.float32(1000.0) + index
    clean = pd.DataFrame(
        {
            "id": ids,
            "well": wells,
            "last_known_tvt": last,
            "pf_ancc": last + np.float32(1.0) + index * np.float32(0.01),
            "beam_mean_d": np.float32(2.0) + index * np.float32(0.02),
            "likpf_mean_d": np.float32(3.0) + index * np.float32(0.03),
            "sc_ens_d": np.float32(4.0) + index * np.float32(0.04),
            "hyb_d": np.float32(5.0) + index * np.float32(0.05),
        }
    )
    formation = pd.DataFrame(
        {
            "id": ids,
            "well": wells,
            "tvt_dense_d": np.float32(6.0) + index * np.float32(0.06),
            "tvt_densew_d": np.float32(7.0) + index * np.float32(0.07),
            "tvt_dense50_d": np.float32(8.0) + index * np.float32(0.08),
        }
    )
    source = pd.DataFrame(
        {
            "id": ids,
            "well": wells,
            module.SOURCE_COMPONENT_COLUMNS[0]: (
                np.float32(0.1) + index * np.float32(0.01)
            ),
            module.SOURCE_COMPONENT_COLUMNS[1]: (
                np.float32(0.2) + index * np.float32(0.01)
            ),
            module.SOURCE_COMPONENT_COLUMNS[2]: (
                np.float32(-0.3) + index * np.float32(0.02)
            ),
        }
    )
    return clean, formation, source


def test_implementation_and_execution_contract_is_fail_closed(module, config) -> None:
    stage0_config = historical_stage0_config(config)
    report = module.validate_scientific_contract(stage0_config)
    assert report["stage"] == "zero_booster_preflight"
    assert report["stage_0_default_phase"] == "train_source"
    assert report["stage_0_split_phases"] == [
        "train_source",
        "train_fold",
        "current_test",
        "aggregate",
    ]
    assert report["parent_features"] == 421
    assert report["added_features"] == 5
    assert report["final_features"] == 426
    assert report["current_execution"] == {
        "models": 0,
        "boosters": 0,
        "predictions": 0,
        "submissions": 0,
    }
    assert report["future_training"] == {
        "variants": 1,
        "lightgbm_configs": 3,
        "folds": 5,
        "gpu_boosters": 15,
        "control_boosters": 0,
    }
    assert report["stage_0_current_test_regeneration"] == {
        "test_wells": 3,
        "pf_ancc_well_runs": 3,
        "pf_z_well_runs": 3,
        "beam_paths": 21,
        "likelihood_pf_well_runs": 3,
        "likelihood_pf_seed_well_trajectories": 384,
        "likelihood_pf_particle_starts": 192000,
    }
    approved = module.validate_scientific_contract(
        stage0_config, require_run_approval=True
    )
    assert approved["stage"] == "zero_booster_preflight"
    identity = module.split_execution_identity(stage0_config)
    assert len(identity["implementation_source_sha256"]) == 64
    assert len(identity["config_sha256"]) == 64
    assert len(identity["scientific_contract_sha256"]) == 64

    unauthorized = copy.deepcopy(stage0_config)
    unauthorized["execution"]["zero_booster_preflight_run_approved"] = False
    with pytest.raises(ValueError, match="canonical adoption lacks Stage 0 approval"):
        module.validate_scientific_contract(unauthorized)

    implementation_only = copy.deepcopy(stage0_config)
    implementation_only["execution"]["current_stage"] = "implementation_only"
    implementation_only["execution"]["zero_booster_preflight_run_approved"] = False
    implementation_only["execution"]["run_preflight"] = False
    implementation_only["execution"]["run_approved"] = False
    implementation_only["preflight"]["enabled"] = False
    implementation_only["implementation"]["canonical_notebook_adopted"] = False
    implementation_only["implementation"]["kaggle_package_created"] = False
    result = module.run_experiment(implementation_only)
    assert result["status"] == "stage_0_implementation_complete_no_execution"
    assert result["kaggle_input_read"] is False
    assert result["boosters_trained"] == 0
    assert result["prediction_rows_generated"] == 0

    audit_root = (
        ROOT
        / "experiments"
        / "exp264_exp263_candidate_confidence_dual_selector"
        / "artifacts"
        / "feature_availability_audit"
    )
    availability = module.verify_availability_contract(
        audit_root / "exp218_feature_availability.csv",
        audit_root / "exp218_clean_273_allowlist.csv",
        stage0_config,
    )
    assert availability["historical_grwr5_values_selected"] == 0
    assert availability["entropy_interaction_selected"] == 0


def test_aggregate_fold4_v2_runtime_alias_is_unambiguous(
    module,
    config,
    tmp_path: Path,
) -> None:
    local_config = copy.deepcopy(config)
    fold_manifest_name = f"{module.OUTPUT_PREFIX}_stage0_train_fold_manifest.json"
    partition_manifest_name = f"{module.OUTPUT_PREFIX}_partition_manifest.csv"
    fold_roots: list[Path] = []
    for outer_fold in range(5):
        root = tmp_path / f"train_fold{outer_fold}" / "artifacts"
        root.mkdir(parents=True)
        (root / fold_manifest_name).touch()
        (root / partition_manifest_name).touch()
        fold_roots.append(root)

    runtime_alias = str(fold_roots[4])
    patterns = local_config["runtime"]["kaggle"]["split_stage_0"]["train_folds"][
        "4"
    ]["artifact_root_patterns"]
    patterns.insert(0, runtime_alias)

    resolved = module.resolve_train_fold_artifact_root(
        local_config,
        outer_fold=4,
        required_files=[fold_manifest_name, partition_manifest_name],
    )
    assert resolved == fold_roots[4].resolve()

    aggregate_source = AGGREGATE.read_text()
    expected_kaggle_alias = (
        "kentookumura/exp402-foldsafe-grwr5-train-fold4-v2/artifacts"
    )
    assert expected_kaggle_alias in aggregate_source
    assert (
        "fold4_artifact_patterns.insert(0, FOLD4_RUNTIME_ARTIFACT_ROOT)"
        in aggregate_source
    )


def test_inference_contract_is_fail_closed(inference_module, config) -> None:
    status = inference_module.validate_inference_is_disabled(config)
    assert status["stage_0_implemented"] is True
    assert status["stage_1_training_implemented"] is True
    assert status["inference_implemented"] is False
    assert status["submission_enabled"] is False
    assert status["run_train"] is True

    leaked = copy.deepcopy(config)
    leaked["execution"]["run_inference"] = True
    with pytest.raises(ValueError, match="inference is forbidden"):
        inference_module.validate_inference_is_disabled(leaked)


def test_stage1_training_contract_is_exact_and_fail_closed(
    stage1_module,
    config,
) -> None:
    report = stage1_module.validate_stage1_contract(config)
    assert report == {
        "active_variants": 1,
        "variant": "fold_safe_grwr_5_addonly",
        "lightgbm_configs": 3,
        "config_indices": [0, 1, 2],
        "folds": 5,
        "planned_gpu_boosters": 15,
        "control_retraining_boosters": 0,
        "inference": False,
        "submission": False,
    }

    retrained_control = copy.deepcopy(config)
    retrained_control["model"]["training"]["control_retraining"] = True
    with pytest.raises(ValueError, match="control retraining is forbidden"):
        stage1_module.validate_stage1_contract(retrained_control)

    unapproved = copy.deepcopy(config)
    unapproved["execution"]["training_run_approved"] = False
    with pytest.raises(PermissionError, match="approval is incomplete"):
        stage1_module.validate_stage1_contract(unapproved)

    missing_exp145 = copy.deepcopy(config)
    missing_exp145["runtime"]["kaggle"]["train_kernel_sources"].remove(
        "kentookumura/exp145-train"
    )
    with pytest.raises(ValueError, match="exactly 11 unique kernel inputs"):
        stage1_module.validate_stage1_contract(missing_exp145)


def test_stage1_artifact_resolver_uses_manifest_sha_not_mount_name(
    stage1_module,
    tmp_path: Path,
) -> None:
    first = tmp_path / "mounted-fold-a" / "artifacts"
    expected = tmp_path / "unexpected-kaggle-mount-name" / "artifacts"
    first.mkdir(parents=True)
    expected.mkdir(parents=True)
    for root, payload in [(first, b"wrong"), (expected, b"expected")]:
        (root / "manifest.json").write_bytes(payload)
        (root / "partition.csv").write_text("fold,role\n0,valid\n")

    resolved = stage1_module.resolve_artifact_root(
        ["/kaggle/input/notebooks/old-slug/artifacts"],
        [tmp_path],
        required_files=["manifest.json", "partition.csv"],
        required_file_sha256={
            "manifest.json": stage1_module.sha256_file(
                expected / "manifest.json"
            ),
            "partition.csv": stage1_module.sha256_file(
                expected / "partition.csv"
            ),
        },
        label="synthetic Stage 1 mount",
    )
    assert resolved == expected.resolve()


def test_grwr5_formula_order_dtype_and_interactions(module) -> None:
    clean, formation, source = make_grwr_inputs(module)
    result = module.build_grwr5_features(clean, formation, source)
    assert list(result.columns) == ["id", "well", *module.GRWR5_FEATURES]
    assert all(result[column].dtype == np.dtype("float32") for column in module.GRWR5_FEATURES)

    last = clean["last_known_tvt"].to_numpy(np.float32)
    candidates = np.vstack(
        [
            clean["pf_ancc"].to_numpy(np.float32),
            last + clean["beam_mean_d"].to_numpy(np.float32),
            last + clean["likpf_mean_d"].to_numpy(np.float32),
            last + clean["sc_ens_d"].to_numpy(np.float32),
            last + clean["hyb_d"].to_numpy(np.float32),
            last + formation["tvt_dense_d"].to_numpy(np.float32),
            last + formation["tvt_densew_d"].to_numpy(np.float32),
            last + formation["tvt_dense50_d"].to_numpy(np.float32),
        ]
    ).astype(np.float32)
    expected_std = np.std(candidates, axis=0, ddof=0).astype(np.float32)
    expected_range = (np.max(candidates, axis=0) - np.min(candidates, axis=0)).astype(
        np.float32
    )
    np.testing.assert_array_equal(
        result[module.GRWR5_FEATURES[0]].to_numpy(np.float32), expected_std
    )
    np.testing.assert_array_equal(
        result[module.GRWR5_FEATURES[1]].to_numpy(np.float32), expected_range
    )
    np.testing.assert_array_equal(
        result[module.GRWR5_FEATURES[2]].to_numpy(np.float32),
        source[module.SOURCE_COMPONENT_COLUMNS[0]].to_numpy(np.float32)
        * expected_std,
    )
    np.testing.assert_array_equal(
        result[module.GRWR5_FEATURES[3]].to_numpy(np.float32),
        source[module.SOURCE_COMPONENT_COLUMNS[1]].to_numpy(np.float32)
        * expected_range,
    )
    np.testing.assert_array_equal(
        result[module.GRWR5_FEATURES[4]].to_numpy(np.float32),
        source[module.SOURCE_COMPONENT_COLUMNS[2]].to_numpy(np.float32)
        * expected_range,
    )


def test_selected_source_components_match_exp218_reference(
    module,
    exp218_module,
    config,
    tmp_path: Path,
) -> None:
    well = "synthetic"
    row_count = 192
    query_rows = np.arange(80, 112, dtype=np.int32)
    md = 1000.0 + np.arange(row_count, dtype=np.float32) * np.float32(0.5)
    gr = (
        75.0
        + 11.0 * np.sin(np.arange(row_count, dtype=np.float32) / 8.0)
        + 3.0 * np.cos(np.arange(row_count, dtype=np.float32) / 19.0)
    ).astype(np.float32)
    tvt_input = np.full(row_count, np.nan, dtype=np.float32)
    tvt_input[:80] = 1500.0 + np.arange(80, dtype=np.float32) * np.float32(0.3)
    pd.DataFrame({"MD": md, "GR": gr, "TVT_input": tvt_input}).to_csv(
        tmp_path / f"{well}__horizontal_well.csv", index=False
    )
    type_tvt = np.linspace(1450.0, 1650.0, 401, dtype=np.float32)
    type_gr = (
        72.0
        + 10.0 * np.sin((type_tvt - 1450.0) / 5.0)
        + 2.0 * np.cos((type_tvt - 1450.0) / 13.0)
    ).astype(np.float32)
    pd.DataFrame({"TVT": type_tvt, "GR": type_gr}).to_csv(
        tmp_path / f"{well}__typewell.csv", index=False
    )

    last = np.full(len(query_rows), tvt_input[79], dtype=np.float32)
    base = pd.DataFrame(
        {
            "id": [f"{well}_{row}" for row in query_rows],
            "well": [well] * len(query_rows),
            "last_known_tvt": last,
            "md_since": md[query_rows] - md[79],
            "pf_ancc": last + np.float32(3.0),
            "beam_mean_d": np.full(len(query_rows), 4.0, dtype=np.float32),
            "likpf_mean_d": np.full(len(query_rows), 5.0, dtype=np.float32),
            "sc_ens_d": np.full(len(query_rows), 6.0, dtype=np.float32),
            "hyb_d": np.full(len(query_rows), 7.0, dtype=np.float32),
            "tvt_dense_d": np.full(len(query_rows), 8.0, dtype=np.float32),
            "tvt_densew_d": np.full(len(query_rows), 9.0, dtype=np.float32),
            "tvt_dense50_d": np.full(len(query_rows), 10.0, dtype=np.float32),
        }
    )
    gr_config, source_evidence = module.validate_formula_source_contract(
        EXP218_SOURCE,
        EXP218_CONFIG,
        config,
    )
    assert source_evidence["whole_grwr_generator_called"] is False
    selected, summary = module.build_grwr_source_components(
        base[["id", "well", "last_known_tvt", "likpf_mean_d"]],
        raw_dir=tmp_path,
        gr_config=gr_config,
    )
    reference, _groups, _summary, _metadata = (
        exp218_module.build_gr_wavelet_rotation_confidence_features(
            base,
            tmp_path,
            gr_config,
        )
    )
    assert len(summary) == 1
    np.testing.assert_allclose(
        selected[module.SOURCE_COMPONENT_COLUMNS].to_numpy(np.float32),
        reference[module.SOURCE_COMPONENT_COLUMNS].to_numpy(np.float32),
        rtol=0.0,
        atol=1e-6,
    )


def test_fold_role_boundary_accepts_train_and_valid_and_rejects_leakage(module) -> None:
    outer_train = pd.DataFrame(
        {"id": ["a_0", "b_0"], "well": ["a", "b"]}
    )
    train_item = {
        "reference_wells": 2,
        "reference_well_sha256": module.sha256_json(["a", "b"]),
        "target_well_sha256": module.sha256_json(["a", "b"]),
        "target_wells_inside_reference": 2,
        "target_wells_self_excluded_from_reference_query": 2,
    }
    train = module.verify_fold_role_boundary(
        train_item,
        expected=outer_train,
        outer_train=outer_train,
        role="train",
    )
    assert train["target_wells_inside_reference"] == 2

    valid_expected = pd.DataFrame({"id": ["c_0"], "well": ["c"]})
    valid_item = {
        "reference_wells": 2,
        "reference_well_sha256": module.sha256_json(["a", "b"]),
        "target_well_sha256": module.sha256_json(["c"]),
        "target_wells_inside_reference": 0,
        "target_wells_self_excluded_from_reference_query": 0,
    }
    valid = module.verify_fold_role_boundary(
        valid_item,
        expected=valid_expected,
        outer_train=outer_train,
        role="valid",
    )
    assert valid["target_wells_inside_reference"] == 0

    leaked = dict(valid_item)
    leaked["target_wells_inside_reference"] = 1
    leaked_train = pd.concat(
        [outer_train, pd.DataFrame({"id": ["c_1"], "well": ["c"]})],
        ignore_index=True,
    )
    leaked["reference_wells"] = 3
    leaked["reference_well_sha256"] = module.sha256_json(["a", "b", "c"])
    with pytest.raises(ValueError, match="leaked"):
        module.verify_fold_role_boundary(
            leaked,
            expected=valid_expected,
            outer_train=leaked_train,
            role="valid",
        )


def test_current_test_frame_contract_and_target_read_guard(module, config) -> None:
    clean, formation, source = make_grwr_inputs(module, rows=6)
    local_config = copy.deepcopy(config)
    local_config["validation"]["expected_current_test_rows"] = 6
    local_config["validation"]["expected_current_test_wells"] = 2
    result, evidence = module.build_current_test_grwr5_from_frames(
        replay_frame=clean,
        source_components=source,
        formation_surface=formation,
        formation_evidence={"target_formation_columns_read": False},
        config=local_config,
    )
    assert len(result) == 6
    assert evidence["wells"] == 2
    assert evidence["model_count"] == 0
    assert evidence["prediction_count"] == 0
    assert evidence["target_formation_columns_read"] is False

    with pytest.raises(ValueError, match="target formation columns"):
        module.build_current_test_grwr5_from_frames(
            replay_frame=clean,
            source_components=source,
            formation_surface=formation,
            formation_evidence={"target_formation_columns_read": True},
            config=local_config,
        )


def test_split_stage_dispatch_is_explicit_and_fail_closed(
    module,
    config,
    tmp_path: Path,
    monkeypatch,
) -> None:
    stage0_config = historical_stage0_config(config)
    calls: list[str] = []

    def fake_current_test(*args, **kwargs):
        calls.append("current_test")
        return {"phase": "current_test", "passed": True}

    def fake_aggregate(*args, **kwargs):
        calls.append("aggregate")
        return {"phase": "aggregate", "passed": True}

    def fake_train_fold(*args, **kwargs):
        calls.append(f"train_fold_{kwargs['outer_fold']}")
        return {
            "phase": "train_fold",
            "outer_fold": kwargs["outer_fold"],
            "passed": True,
        }

    monkeypatch.setattr(module, "run_stage0_current_test", fake_current_test)
    monkeypatch.setattr(module, "run_stage0_aggregate", fake_aggregate)
    monkeypatch.setattr(module, "run_stage0_train_fold", fake_train_fold)
    fold = module.run_zero_booster_preflight(
        stage0_config,
        output_dir=tmp_path,
        phase="train_fold",
        outer_fold=3,
    )
    current = module.run_zero_booster_preflight(
        stage0_config,
        output_dir=tmp_path,
        phase="current_test",
    )
    aggregate = module.run_zero_booster_preflight(
        stage0_config,
        output_dir=tmp_path,
        phase="aggregate",
    )
    assert fold == {"phase": "train_fold", "outer_fold": 3, "passed": True}
    assert current == {"phase": "current_test", "passed": True}
    assert aggregate == {"phase": "aggregate", "passed": True}
    assert calls == ["train_fold_3", "current_test", "aggregate"]

    with pytest.raises(ValueError, match="requires outer_fold"):
        module.run_zero_booster_preflight(
            stage0_config,
            output_dir=tmp_path,
            phase="train_fold",
        )

    with pytest.raises(ValueError, match="requires one split phase"):
        module.run_zero_booster_preflight(
            stage0_config,
            output_dir=tmp_path,
            phase="monolithic",
        )


def test_split_aggregate_verifies_upstream_files(
    module,
    config,
    tmp_path: Path,
    monkeypatch,
) -> None:
    local_config = historical_stage0_config(config)
    local_config["validation"]["expected_rows"] = 2
    source_root = tmp_path / "train_source"
    current_root = tmp_path / "current_test"
    output_root = tmp_path / "aggregate"
    source_root.mkdir()
    current_root.mkdir()
    output_root.mkdir()
    identity = module.split_execution_identity(local_config)

    source_path = (
        source_root / f"{module.OUTPUT_PREFIX}_train_context_source.parquet"
    )
    source_path.write_bytes(b"source-components")
    source_manifest = {
        "passed": True,
        "execution_identity": identity,
        "generated": {
            "train_context_source": {
                "path": source_path.name,
                "file_sha256": module.sha256_file(source_path),
                "logical_content_sha256": "5" * 64,
            },
        },
        "models_trained": 0,
        "boosters_trained": 0,
    }
    module.write_json(
        source_root
        / f"{module.OUTPUT_PREFIX}_stage0_train_source_manifest.json",
        source_manifest,
    )

    fold_roots = {}
    for outer_fold in range(5):
        fold_root = tmp_path / f"train_fold{outer_fold}"
        fold_root.mkdir()
        fold_roots[outer_fold] = fold_root
        partition_rows = []
        for role in ["train", "valid"]:
            relative = (
                Path("fold_safe_grwr5")
                / f"downstream_outer_fold={outer_fold}"
                / f"role={role}"
                / "part-00000.parquet"
            )
            path = fold_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"{outer_fold}:{role}".encode())
            partition_rows.append(
                {
                    "downstream_outer_fold": outer_fold,
                    "role": role,
                    "rows": 1,
                    "wells": 1,
                    "path": str(relative),
                    "file_sha256": module.sha256_file(path),
                    "row_identity_sha256": "1" * 64,
                    "grwr5_schema_sha256": module.sha256_json(
                        module.GRWR5_FEATURES
                    ),
                    "grwr5_logical_content_sha256": "2" * 64,
                    "formation_logical_content_sha256": "3" * 64,
                    "formation_reference_wells": 1,
                    "formation_target_wells_inside_reference": 0,
                    "formation_target_columns_read": False,
                    "historical_grwr5_values_loaded": 0,
                    "entropy_or_score_features_loaded": 0,
                }
            )
        partition_manifest_path = (
            fold_root / f"{module.OUTPUT_PREFIX}_partition_manifest.csv"
        )
        pd.DataFrame(partition_rows).to_csv(
            partition_manifest_path, index=False
        )
        fold_manifest = {
            "passed": True,
            "outer_fold": outer_fold,
            "execution_identity": identity,
            "feature_schema_sha256": {"parent_421": "4" * 64},
            "generated": {
                "outer_role_partition_manifest": {
                    "path": partition_manifest_path.name,
                    "sha256": module.sha256_file(partition_manifest_path),
                    "partitions": 2,
                },
            },
            "models_trained": 0,
            "boosters_trained": 0,
        }
        module.write_json(
            fold_root
            / f"{module.OUTPUT_PREFIX}_stage0_train_fold_manifest.json",
            fold_manifest,
        )

    current_path = current_root / f"{module.OUTPUT_PREFIX}_current_test_grwr5.parquet"
    current_path.write_bytes(b"current-test")
    current_manifest = {
        "passed": True,
        "execution_identity": identity,
        "generated": {
            "current_test": {
                "path": current_path.name,
                "file_sha256": module.sha256_file(current_path),
                "logical_content_sha256": "6" * 64,
                "rows": local_config["validation"][
                    "expected_current_test_rows"
                ],
                "wells": local_config["validation"][
                    "expected_current_test_wells"
                ],
                "target_formation_columns_read": False,
            }
        },
        "models_trained": 0,
        "boosters_trained": 0,
    }
    module.write_json(
        current_root
        / f"{module.OUTPUT_PREFIX}_stage0_current_test_manifest.json",
        current_manifest,
    )

    def fake_resolve(_config, *, phase, required_files):
        del required_files
        return source_root if phase == "train_source" else current_root

    def fake_fold_resolve(_config, *, outer_fold, required_files):
        del required_files
        return fold_roots[outer_fold]

    monkeypatch.setattr(module, "resolve_split_artifact_root", fake_resolve)
    monkeypatch.setattr(
        module,
        "resolve_train_fold_artifact_root",
        fake_fold_resolve,
    )
    result = module.run_stage0_aggregate(
        local_config,
        output_dir=output_root,
    )
    assert result["passed"] is True
    assert all(result["checks"].values())
    assert (
        output_root / f"{module.OUTPUT_PREFIX}_preflight_manifest.json"
    ).is_file()
    assert (
        output_root / f"{module.OUTPUT_PREFIX}_reproducibility_manifest.json"
    ).is_file()


def test_historical_outputs_and_score_columns_are_rejected(module) -> None:
    clean, formation, source = make_grwr_inputs(module)
    historical = clean.assign(grwr_candidate_tvt_std=np.float32(1.0))
    with pytest.raises(ValueError, match="historical GRWR-5"):
        module.build_grwr5_features(historical, formation, source)

    score_leak = clean.assign(learned_prob_margin=np.float32(0.5))
    with pytest.raises(ValueError, match="exp111 score"):
        module.build_grwr5_features(score_leak, formation, source)


def test_candidate_sources_are_not_canonical_and_contain_no_model_training() -> None:
    train_source = TRAIN.read_text()
    stage1_source = STAGE1_TRAIN.read_text()
    inference_source = INFERENCE.read_text()
    assert "__file__" not in train_source
    assert "importlib" not in train_source
    assert "spec_from_file_location" not in train_source
    assert "build_gr_wavelet_rotation_confidence_features(" not in train_source
    assert "import lightgbm" not in train_source
    assert ".fit(" not in train_source
    assert "submission_path" not in train_source
    assert '["id", "prediction"]' not in train_source
    assert "__file__" not in stage1_source
    assert "from lightgbm import LGBMRegressor" in stage1_source
    assert ".fit(" in stage1_source
    assert "planned_gpu_boosters" in stage1_source
    assert "control_retraining_boosters" in stage1_source
    assert "promotion PASS" in inference_source
    assert (EXP_DIR / f"{EXP_NAME}_train.ipynb").exists()
    assert (EXP_DIR / f"{EXP_NAME}_inference.ipynb").exists()
