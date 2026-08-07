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

ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "experiments" / "exp333_exp226_k16_segment_residual_offset_target"
TRAIN_SOURCE = (
    EXP_DIR
    / "exp333_exp226_k16_segment_residual_offset_target_stage1_compact_selfcontained_train.py"
)
INFERENCE_SOURCE = (
    EXP_DIR
    / "exp333_exp226_k16_segment_residual_offset_target_compact_selfcontained_inference.py"
)
CANDIDATE_INFERENCE_SOURCE = (
    EXP_DIR
    / "exp333_exp226_k16_segment_residual_offset_target_current_test_compact_selfcontained_inference.py"
)


def load_module(path: Path, name: str):
    previous = os.environ.get("EXP333_IMPORT_ONLY")
    os.environ["EXP333_IMPORT_ONLY"] = "1"
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        if previous is None:
            os.environ.pop("EXP333_IMPORT_ONLY", None)
        else:
            os.environ["EXP333_IMPORT_ONLY"] = previous


train = load_module(TRAIN_SOURCE, "exp333_train")
inference = load_module(INFERENCE_SOURCE, "exp333_inference")
candidate_inference = load_module(
    CANDIDATE_INFERENCE_SOURCE, "exp333_candidate_inference"
)


@pytest.fixture
def config() -> dict:
    value = yaml.safe_load((EXP_DIR / "config.yaml").read_text())
    assert isinstance(value, dict)
    return value


def synthetic_target_free(wells: int = 2, rows_per_well: int = 32) -> pd.DataFrame:
    records: list[dict] = []
    for well in range(wells):
        for offset in range(rows_per_well):
            records.append(
                {
                    "well_id": f"well_{well}",
                    "row_idx": 1000 * well + offset,
                    "suffix_offset": offset,
                    "tvt_pred": 12000.0 + 10.0 * well + offset,
                    "fold": well,
                }
            )
    return pd.DataFrame(records, columns=train.TARGET_FREE_COLUMNS)


def test_config_records_failed_full_stage1_train_fail_closed(config: dict) -> None:
    contract = train.validate_scientific_contract(
        config, require_execution_authorization=False
    )
    assert config["experiment"]["route"] == "ensemble"
    assert (
        config["experiment"]["status"]
        == "current_test_candidate_inference_v2_completed_no_submission"
    )
    assert config["implementation"] == {
        "enabled": True,
        "scope": "stage_0_and_stage_1_train",
        "stage_0_enabled": True,
        "stage_1_enabled": True,
        "inference_enabled": False,
        "submission_enabled": False,
    }
    assert (
        config["execution_contract"]["selected_stage"]
        == "stage_1_train_completed_fail_closed"
    )
    assert config["execution_contract"]["stage_0"] == {
        "readouts": 1,
        "variants": 0,
        "model_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
    }
    assert contract["execution_authorization"] == {
        "selected_stage": "stage_1_train_completed_fail_closed",
        "kaggle_push_approved": False,
        "stage_0_run_approved": False,
    }
    with pytest.raises(RuntimeError, match="not authorized"):
        train.validate_scientific_contract(
            config, require_execution_authorization=True
        )
    assert config["execution_contract"]["stage_0_run_authorization_consumed"] is True
    assert config["stage_0_headroom"]["result"]["decision"] == "PASS_STAGE0"
    assert config["stage_0_headroom"]["result"]["improved_folds"] == 5
    assert config["implementation"]["stage_1_enabled"] is True
    assert config["execution_contract"]["stage_1_preflight_approved"] is False
    assert (
        config["execution_contract"]["stage_1_preflight_authorization_consumed"]
        is True
    )
    assert config["execution_contract"]["stage_1_run_approved"] is False
    assert config["execution_contract"]["stage_1_run_authorization_consumed"] is True
    assert config["runtime"]["kaggle"]["train_kernel_sources"] == [
        "kentookumura/exp072-exp063-full-replay-feature-cache-train",
        "kentookumura/exp226-k16-kappa-repro-train"
    ]
    train.validate_stage1_contract(config)
    with pytest.raises(RuntimeError, match="preflight execution is not authorized"):
        train.validate_stage1_contract(config, execution_mode="preflight")
    with pytest.raises(RuntimeError, match="train execution is not authorized"):
        train.validate_stage1_contract(config, execution_mode="train")


def test_frozen_contract_rejects_stage1_disable_or_target_change(config: dict) -> None:
    changed = copy.deepcopy(config)
    changed["implementation"]["stage_1_enabled"] = False
    with pytest.raises(ValueError, match="frozen Stage 0/1 contract changed"):
        train.validate_scientific_contract(
            changed, require_execution_authorization=False
        )
    changed = copy.deepcopy(config)
    changed["target"]["aggregation"] = "median"
    with pytest.raises(ValueError, match="frozen Stage 0/1 contract changed"):
        train.validate_scientific_contract(
            changed, require_execution_authorization=False
        )


def test_exact_k16_assignment_matches_parent_formula() -> None:
    for length in (16, 17, 32, 101, 5108):
        edges = np.linspace(0.0, float(length), 17)
        expected = np.clip(
            np.searchsorted(
                edges[1:], np.arange(1, length + 1, dtype=np.float64), side="left"
            ),
            0,
            15,
        ).astype(np.int16)
        actual = train.exact_k16_segment_ids(length)
        np.testing.assert_array_equal(actual, expected)
        assert int(actual.min()) == 0
        assert int(actual.max()) == 15
    with pytest.raises(ValueError, match="fixed to K16"):
        train.exact_k16_segment_ids(32, k_segments=12)


def test_target_free_validation_and_segment_coverage(config: dict) -> None:
    safe = train.validate_target_free_rows(
        synthetic_target_free(), config, enforce_expected_counts=False
    )
    assigned, counts = train.assign_k16_segments(
        safe, config, enforce_expected_counts=False
    )
    assert len(assigned) == 64
    assert len(counts) == 32
    assert counts["segment_row_count"].eq(2).all()
    assert assigned.groupby(["well_id", "segment_id"]).size().eq(2).all()
    broken = safe.copy()
    broken.loc[1, "suffix_offset"] = 8
    with pytest.raises(ValueError, match="not contiguous"):
        train.validate_target_free_rows(
            broken, config, enforce_expected_counts=False
        )


def test_truth_is_attached_only_after_nonempty_freeze_sha(config: dict) -> None:
    safe = train.validate_target_free_rows(
        synthetic_target_free(), config, enforce_expected_counts=False
    )
    assigned, counts = train.assign_k16_segments(
        safe, config, enforce_expected_counts=False
    )
    evidence = {
        "decompressed_sha256": "input",
    }
    freeze = train.build_target_free_freeze(assigned, counts, evidence, config)
    assert freeze["truth_columns_loaded_before_freeze"] == 0
    assert len(freeze["target_free_contract_sha256"]) == 64
    truth = assigned[["well_id", "row_idx"]].copy()
    truth["tvt_true"] = assigned["tvt_pred"] + 1.0
    with pytest.raises(ValueError, match="frozen target-free contract"):
        train.attach_truth_after_freeze(
            assigned, truth, target_free_contract_sha256=""
        )
    joined = train.attach_truth_after_freeze(
        assigned,
        truth,
        target_free_contract_sha256=freeze["target_free_contract_sha256"],
    )
    assert joined["tvt_true"].notna().all()


def test_segment_mean_is_optimal_constant_offset(config: dict) -> None:
    safe = train.validate_target_free_rows(
        synthetic_target_free(), config, enforce_expected_counts=False
    )
    assigned, _ = train.assign_k16_segments(
        safe, config, enforce_expected_counts=False
    )
    truth = assigned[["well_id", "row_idx"]].copy()
    residual = np.tile([1.0, 3.0], len(assigned) // 2)
    truth["tvt_true"] = assigned["tvt_pred"].to_numpy() + residual
    joined = train.attach_truth_after_freeze(
        assigned, truth, target_free_contract_sha256="frozen"
    )
    synthetic_config = copy.deepcopy(config)
    synthetic_config["data"]["exp226_oof"]["expected_rmse"] = float(np.sqrt(5.0))
    readout = train.build_oracle_readout(
        joined, synthetic_config, enforce_expected_counts=False
    )
    np.testing.assert_allclose(readout.segment_table["segment_mean_residual"], 2.0)
    assert train.rmse(
        truth["tvt_true"].to_numpy(), readout.base_prediction
    ) == pytest.approx(np.sqrt(5.0))
    assert train.rmse(
        truth["tvt_true"].to_numpy(), readout.oracle_prediction
    ) == pytest.approx(1.0)
    assert readout.summary["oracle_offset_persisted"] is False
    assert readout.summary["oracle_prediction_persisted"] is False
    assert readout.summary["segment_target_persisted"] is False
    assert readout.summary["technical_pass"] is True


def test_stage0_gate_requires_overall_and_five_of_five_folds(config: dict) -> None:
    passed = train.evaluate_stage0_gate(1.01, [0.5] * 5, config)
    assert passed["decision"] == "PASS_STAGE0"
    assert passed["stage_1_may_be_implemented"] is True
    failed = train.evaluate_stage0_gate(1.01, [0.5, 0.5, 0.49, 0.5, 0.5], config)
    assert failed["decision"] == "FAIL_CLOSE_BRANCH"
    assert failed["stage_1_may_be_implemented"] is False
    failed = train.evaluate_stage0_gate(0.99, [0.6] * 5, config)
    assert failed["decision"] == "FAIL_CLOSE_BRANCH"


def test_content_sha_detects_assignment_mutation(config: dict) -> None:
    safe = train.validate_target_free_rows(
        synthetic_target_free(), config, enforce_expected_counts=False
    )
    assigned, counts = train.assign_k16_segments(
        safe, config, enforce_expected_counts=False
    )
    before = train.build_target_free_freeze(
        assigned, counts, {"decompressed_sha256": "input"}, config
    )["segment_assignment_sha256"]
    mutated = assigned.copy()
    mutated.loc[0, "segment_id"] = 1
    after = train.frame_content_sha256(
        mutated,
        ("well_id", "row_idx", "suffix_offset", "fold", "segment_id"),
    )
    assert after != before


def test_inference_is_fail_closed(config: dict) -> None:
    report = inference.validate_disabled_inference(config)
    assert report["status"] == "disabled_fail_closed"
    assert report["deployable_model_available"] is False
    assert report["submission_created"] is False
    changed = copy.deepcopy(config)
    changed["inference"]["enabled"] = True
    with pytest.raises(ValueError, match="must remain disabled"):
        inference.validate_disabled_inference(changed)


def test_candidate_inference_is_separately_authorized_without_submission(
    config: dict,
) -> None:
    contract = candidate_inference.validate_candidate_inference_contract(config)
    assert contract == {
        "authorization_scope": "current_test_candidate_artifact_only",
        "variant_count": 1,
        "saved_model_inference_count": 5,
        "trained_model_configs": 0,
        "trained_folds": 0,
        "trained_boosters": 0,
        "parent_control_retraining": False,
        "submission_created": False,
    }
    assert config["implementation"]["inference_enabled"] is False
    assert config["inference"]["enabled"] is False
    assert config["candidate_inference"]["enabled"] is True
    assert config["candidate_inference"]["create_submission"] is False
    assert config["execution_contract"]["candidate_inference_approved"] is False
    assert (
        config["execution_contract"]["candidate_inference_authorization_consumed"]
        is True
    )
    with pytest.raises(RuntimeError, match="execution is not authorized"):
        candidate_inference.validate_candidate_inference_contract(
            config, require_execution_authorization=True
        )
    changed = copy.deepcopy(config)
    changed["candidate_inference"]["offset_shrinkage"] = "clip"
    with pytest.raises(ValueError, match="candidate inference contract changed"):
        candidate_inference.validate_candidate_inference_contract(changed)


def test_candidate_exact_k16_and_target_free_segment_aggregation() -> None:
    np.testing.assert_array_equal(
        candidate_inference.exact_k16_segment_ids(32),
        train.exact_k16_segment_ids(32),
    )
    rows = pd.DataFrame(
        {
            "well_id": np.repeat(["well_a", "well_b", "well_c"], 32),
            "row_idx": np.tile(np.arange(32), 3),
            "suffix_offset": np.tile(np.arange(32), 3),
            "segment_id": np.tile(
                candidate_inference.exact_k16_segment_ids(32), 3
            ),
            "md_since": np.tile(np.arange(32, dtype=float), 3) * 10.0,
            "exp226_tvt": 12000.0 + np.tile(np.arange(32, dtype=float), 3),
            "feature_mean": np.tile([1.0, 3.0], 48),
            "feature_all_nan": np.nan,
        }
    )
    segments = candidate_inference.aggregate_current_test_segments(
        rows, ("feature_mean", "feature_all_nan")
    )
    assert len(segments) == 48
    assert segments["segment_row_count"].eq(2).all()
    np.testing.assert_allclose(segments["feature_mean"], 2.0)
    assert segments["feature_all_nan"].isna().all()
    np.testing.assert_allclose(segments["segment_md_span"], 10.0)
    np.testing.assert_allclose(segments["exp226_pred_end_minus_start"], 1.0)


def test_stable_inner_fold_manifest_is_balanced_and_outer_specific() -> None:
    wells = [f"well_{index:02d}" for index in range(17)]
    first = train.stable_inner_fold_manifest(wells, outer_fold=0)
    repeated = train.stable_inner_fold_manifest(list(reversed(wells)), outer_fold=0)
    pd.testing.assert_frame_equal(first, repeated)
    counts = first["inner_fold"].value_counts()
    assert counts.max() - counts.min() <= 1
    second_outer = train.stable_inner_fold_manifest(wells, outer_fold=1)
    assert first["inner_digest"].tolist() != second_outer["inner_digest"].tolist()


def test_exp072_loader_never_reads_target_and_checks_schema(
    config: dict, tmp_path: Path
) -> None:
    changed = copy.deepcopy(config)
    feature_columns = [f"feature_{index:03d}" for index in range(196)]
    cache = pd.DataFrame(
        {
            "id": ["well_0_0", "well_0_1"],
            "well": ["well_0", "well_0"],
            "target": [999.0, 999.0],
            **{
                column: [float(index), float(index + 1)]
                for index, column in enumerate(feature_columns)
            },
        }
    )
    cache_path = tmp_path / "cache.csv.gz"
    cache.to_csv(cache_path, index=False, compression="gzip")
    schema_path = tmp_path / "schema.csv"
    pd.DataFrame(
        {
            "variant": "synthetic",
            "feature_index": np.arange(196),
            "feature": feature_columns,
        }
    ).to_csv(schema_path, index=False)
    changed["data"]["exp072_feature_cache"]["schema_filename"] = schema_path.name
    changed["data"]["exp072_feature_cache"]["schema_patterns"] = [str(schema_path)]
    frame, loaded_features, metadata = train.load_target_free_exp072_frame(
        cache_path, changed
    )
    assert "target" not in frame.columns
    assert loaded_features == feature_columns
    assert metadata["target_columns_loaded"] == 0
    assert metadata["schema_file_sha256"] == train.sha256_file(schema_path)


def test_segment_aggregation_uses_float64_finite_mean_and_fixed_structure() -> None:
    safe = synthetic_target_free(wells=2, rows_per_well=32)
    assigned, _ = train.assign_k16_segments(
        safe, {}, enforce_expected_counts=False
    )
    nested = assigned.rename(columns={"fold": "outer_fold"}).copy()
    nested["role"] = "outer_valid"
    nested["inner_fold"] = -1
    surface = nested[["well_id", "row_idx"]].copy()
    surface["md_since"] = nested["suffix_offset"].astype(float) * 10.0
    surface["feature_mean"] = np.tile([1.0, 3.0], len(surface) // 2)
    surface["feature_all_nan"] = np.nan
    truth = nested[["well_id", "row_idx"]].copy()
    truth["tvt_true"] = nested["tvt_pred"].to_numpy() + np.tile(
        [1.0, 3.0], len(nested) // 2
    )
    segments = train.aggregate_stage1_segments(
        nested,
        surface,
        truth,
        ("feature_mean", "feature_all_nan"),
    )
    assert len(segments) == 32
    assert segments["segment_row_count"].eq(2).all()
    np.testing.assert_allclose(segments["feature_mean"], 2.0)
    assert segments["feature_all_nan"].isna().all()
    np.testing.assert_allclose(segments["segment_mean_residual"], 2.0)
    np.testing.assert_allclose(segments["segment_md_span"], 10.0)
    np.testing.assert_allclose(segments["exp226_pred_end_minus_start"], 1.0)


def test_stage1_gate_requires_every_fixed_scope(config: dict) -> None:
    synthetic_config = copy.deepcopy(config)
    gates = synthetic_config["promotion_gates"]
    gates["maximum_pooled_rmse"] = 1.1
    gates["minimum_rmse_gain_vs_exp226_ft"] = 0.2
    gates["inference_candidate_additional_maximum_rmse"] = 1.1
    row_records = []
    segment_records = []
    hidden_records = []
    for fold in range(5):
        well_id = f"well_{fold}"
        hidden_records.append(
            {
                "well_id": well_id,
                "verification_like_spatial_role": "valid",
                "verification_like_typewell_purged_role": "valid",
            }
        )
        for md_since in (100.0, 1200.0):
            row_records.append(
                {
                    "outer_fold": fold,
                    "well_id": well_id,
                    "tvt_true": 10.0,
                    "tvt_pred": 8.0,
                    "tvt_pred_stage1": 9.0,
                    "md_since": md_since,
                    "boundary_band_pm8": True,
                }
            )
        segment_records.append(
            {
                "outer_fold": fold,
                "segment_row_count": 2,
                "segment_mean_residual": 2.0,
                "segment_offset_pred": 1.0,
            }
        )
    evaluation = train.evaluate_stage1_outputs(
        pd.DataFrame(row_records),
        pd.DataFrame(segment_records),
        pd.DataFrame(hidden_records),
        synthetic_config,
    )
    assert evaluation["scientific_pass"] is True
    assert evaluation["inference_candidate_threshold_pass"] is True
    assert evaluation["decision"] == "PASS_STAGE1"
    broken = pd.DataFrame(row_records)
    broken.loc[broken["md_since"].ge(1000.0), "tvt_pred_stage1"] = 7.0
    failed = train.evaluate_stage1_outputs(
        broken,
        pd.DataFrame(segment_records),
        pd.DataFrame(hidden_records),
        synthetic_config,
    )
    assert failed["scientific_pass"] is False
    assert failed["gate_checks"]["1000_plus_nonworse"] is False


def test_compact_sources_are_notebook_safe_and_execution_is_closed(config: dict) -> None:
    train_source = TRAIN_SOURCE.read_text()
    inference_source = INFERENCE_SOURCE.read_text()
    assert "__file__" not in train_source
    assert "__file__" not in inference_source
    assert "# ## Contents" in train_source
    assert "run_stage0_experiment(CONFIG)" in train_source
    assert "run_stage1_experiment(CONFIG)" in train_source
    assert "generate_strict_nested_predictions" in train_source
    assert "LGBMRegressor" in train_source
    assert (
        config["execution_contract"]["selected_stage"]
        == "stage_1_train_completed_fail_closed"
    )
    assert config["execution_contract"]["kaggle_push_approved"] is False
    assert config["execution_contract"]["stage_1_preflight_approved"] is False
    assert config["execution_contract"]["stage_1_run_approved"] is False
    assert config["artifacts"]["expected_inference"] == []
    assert not any(
        "stage0_segment_target.csv" in name or "oracle_prediction" in name
        for name in config["artifacts"]["expected_train"]
    )
