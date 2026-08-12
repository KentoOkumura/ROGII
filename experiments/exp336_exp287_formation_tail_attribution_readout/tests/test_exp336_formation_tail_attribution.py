from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[3]
TRAIN_SOURCE = (
    ROOT
    / "experiments"
    / "exp336_exp287_formation_tail_attribution_readout"
    / "exp336_exp287_formation_tail_attribution_readout_compact_selfcontained_train.py"
)
INFERENCE_SOURCE = (
    ROOT
    / "experiments"
    / "exp336_exp287_formation_tail_attribution_readout"
    / "exp336_exp287_formation_tail_attribution_readout_compact_selfcontained_inference.py"
)


def load_source(path: Path, module_name: str):
    os.environ["EXP336_IMPORT_ONLY"] = "1"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


exp336 = load_source(TRAIN_SOURCE, "exp336_train_test_module")
exp336_inference = load_source(INFERENCE_SOURCE, "exp336_inference_test_module")


def test_config_preserves_zero_booster_and_stage_approval_contract() -> None:
    contract = exp336.validate_scientific_contract(exp336.CONFIG, require_execution=False)
    assert contract["primary_families"] == 6
    assert contract["boosters"] == 0
    execution = exp336.CONFIG["execution"]
    assert contract["stage"] in execution["allowed_stages"]
    if contract["stage"] == "implementation_complete_no_run":
        assert execution["kaggle_push_approved"] is False
        assert execution["run_stage_a_freeze"] is False
        assert execution["run_stage_b_attribution"] is False
    else:
        assert contract["stage"] == "full_attribution_readout"
        assert execution["kaggle_push_approved"] is True
        assert execution["run_stage_a_freeze"] is True
        assert execution["run_stage_b_attribution"] is True
    assert execution["run_inference"] is False
    assert execution["create_submission"] is False


def test_inference_contract_is_fail_closed() -> None:
    contract = exp336_inference.validate_zero_inference_contract(exp336_inference.CONFIG)
    assert contract["boosters"] == 0
    assert contract["run_inference"] is False
    assert contract["create_submission"] is False
    assert contract["submit_to_kaggle"] is False


def test_sha_resolver_accepts_equivalent_copies_in_pattern_order(tmp_path: Path) -> None:
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    different = tmp_path / "different.csv"
    first.write_text("same\n")
    second.write_text("same\n")
    different.write_text("different\n")
    expected = exp336.sha256_file(first)
    selected = exp336.select_first_sha_matched_file(
        [first, second, different],
        expected_sha256=expected,
        label="synthetic equivalent input",
    )
    assert selected == first.resolve()
    with pytest.raises(FileNotFoundError, match="no SHA-matched"):
        exp336.select_first_sha_matched_file(
            [different],
            expected_sha256=expected,
            label="synthetic missing input",
        )


def test_stage_a_rejects_truth_prediction_and_error_columns() -> None:
    for forbidden in ["TVT", "actual_tvt", "candidate__pred_tvt", "abs_error"]:
        with pytest.raises(ValueError, match="forbidden"):
            exp336.validate_stage_a_source_columns(["id", "well", forbidden])
    exp336.validate_stage_a_source_columns(
        ["id", "well", "frm_rmse_ANCC", "dense_rmse", "spatial_knn_dist"]
    )


def synthetic_formation_partition() -> pd.DataFrame:
    wells = np.repeat(["well_a", "well_b"], 8)
    rows = len(wells)
    frame = pd.DataFrame(
        {
            "id": [f"id_{index}" for index in range(rows)],
            "well": wells,
        }
    )
    for index, name in enumerate(exp336.canonical_formation_feature_names()):
        frame[name] = np.full(rows, 1.0 + index / 100.0, dtype=np.float32)
    frame["spatial_knn_dist"] = np.tile(np.arange(8, dtype=np.float32), 2)
    frame["dense_dist"] = np.tile(np.arange(8, dtype=np.float32) + 10, 2)
    frame["dense_std"] = np.tile(np.arange(8, dtype=np.float32) + 20, 2)
    frame["spatial_vs_dense"] = np.tile(np.arange(-4, 4, dtype=np.float32), 2)
    frame["form_rng_d"] = np.tile(np.arange(8, dtype=np.float32) + 30, 2)
    frame["sig_std"] = np.tile(np.arange(8, dtype=np.float32) + 40, 2)
    return frame


def test_stage_a_family_aggregation_uses_fixed_numpy_linear_rules() -> None:
    frame = synthetic_formation_partition()
    aggregated = exp336.aggregate_partition_target_free(
        frame,
        outer_fold=2,
        families=exp336.expected_family_contract(),
    )
    assert aggregated.shape[0] == 2
    expected_p90 = float(np.quantile(np.arange(8), 0.90, method="linear"))
    assert np.allclose(aggregated["plane_reference_distance"], expected_p90)
    expected_abs_p90 = float(np.quantile(np.abs(np.arange(-4, 4)), 0.90, method="linear"))
    assert np.allclose(aggregated["plane_dense_disagreement"], expected_abs_p90)
    calibration_columns = exp336.expected_family_contract()[-1]["source_columns"]
    expected_calibration = max(float(frame[name].iloc[0]) for name in calibration_columns)
    assert np.allclose(aggregated["known_prefix_formation_calibration_error"], expected_calibration)
    assert aggregated["outer_fold"].eq(2).all()


def test_known_prefix_calibration_must_be_well_constant() -> None:
    frame = synthetic_formation_partition()
    frame.loc[1, "frm_rmse_ANCC"] += 0.01
    with pytest.raises(ValueError, match="not constant"):
        exp336.aggregate_partition_target_free(
            frame,
            outer_fold=0,
            families=exp336.expected_family_contract(),
        )


def test_raw_context_uses_only_fixed_prefix_suffix_geometry() -> None:
    frame = pd.DataFrame(
        {
            "MD": [0.0, 1.0, 2.0, 3.0, 4.0],
            "X": [0.0, 1.0, 2.0, 3.0, 4.0],
            "Y": [0.0, 0.0, 0.0, 0.0, 0.0],
            "Z": [0.0, -1.0, -2.0, -3.0, -4.0],
            "TVT_input": [10.0, 11.0, 12.0, np.nan, np.nan],
        }
    )
    context = exp336.compute_raw_context_for_well(frame, denominator_floor=1.0e-6)
    assert context["known_prefix_row_count"] == 3
    assert context["evaluation_row_count"] == 2
    assert context["suffix_to_prefix_md_span_ratio"] == pytest.approx(1.0)
    assert context[
        "evaluation_xy_distance_from_last_known_p90_div_prefix_xy_span"
    ] == pytest.approx(0.95)


def synthetic_target_free_attributes(wells: int = 40) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "well": [f"well_{index:03d}" for index in range(wells)],
            "outer_fold": np.arange(wells) % 5,
        }
    )
    for offset, family in enumerate(exp336.PRIMARY_FAMILY_ORDER):
        frame[family] = np.arange(wells, dtype=np.float64) + offset / 10.0
    frame["signal_disagreement_sig_std_p90"] = np.linspace(1.0, 2.0, wells)
    frame["dense_known_neighbor_std_dense_nb_std"] = np.linspace(2.0, 3.0, wells)
    frame["known_prefix_row_count"] = 50
    frame["evaluation_row_count"] = 100
    frame["evaluation_to_known_row_count_ratio"] = 2.0
    frame["suffix_to_prefix_md_span_ratio"] = 3.0
    frame["evaluation_xy_distance_from_last_known_p90_div_prefix_xy_span"] = 4.0
    frame["prefix_md_span_denominator_floored"] = 0
    frame["prefix_xy_span_denominator_floored"] = 0
    return frame


def config_for_synthetic_wells(wells: int = 40) -> dict:
    config = copy.deepcopy(exp336.CONFIG)
    config["validation"]["expected_wells"] = wells
    gate = config["audit"]["decision_gate"]
    gate["minimum_global_wells_per_endpoint_quartile"] = wells // 4
    gate["minimum_wells_per_endpoint_quartile_per_fold"] = wells // 20
    gate["minimum_wells_per_endpoint_quartile_per_hidden_like_scope"] = wells // 4
    return config


def test_freeze_barrier_detects_attribute_tampering(tmp_path: Path) -> None:
    config = config_for_synthetic_wells()
    attributes, freeze, freeze_sha = exp336.freeze_target_free_attributes(
        synthetic_target_free_attributes(),
        config=config,
        output_dir=tmp_path,
        partition_evidence=[],
        raw_context_audit={"forbidden_value_columns_opened": []},
    )
    assert all(row["eligible"] for row in freeze["families"])
    assert set(attributes[f"{exp336.PRIMARY_FAMILY_ORDER[0]}__risk_quartile"]) == {
        1,
        2,
        3,
        4,
    }
    loaded, _ = exp336.load_and_validate_freeze(
        tmp_path, expected_freeze_manifest_sha256=freeze_sha
    )
    assert len(loaded) == 40
    attribute_path = tmp_path / "target_free_well_attributes.csv"
    attribute_path.write_text(attribute_path.read_text().replace("well_000", "well_bad", 1))
    with pytest.raises(ValueError, match="SHA mismatch"):
        exp336.load_and_validate_freeze(tmp_path, expected_freeze_manifest_sha256=freeze_sha)


def test_non_strict_quartile_edges_make_family_ineligible(tmp_path: Path) -> None:
    config = config_for_synthetic_wells()
    attributes = synthetic_target_free_attributes()
    attributes["plane_reference_distance"] = 1.0
    frozen, freeze, _ = exp336.freeze_target_free_attributes(
        attributes,
        config=config,
        output_dir=tmp_path,
        partition_evidence=[],
        raw_context_audit={"forbidden_value_columns_opened": []},
    )
    family = next(row for row in freeze["families"] if row["name"] == "plane_reference_distance")
    assert family["eligible"] is False
    assert frozen["plane_reference_distance__risk_quartile"].eq(0).all()

    well = frozen.copy()
    well["rows"] = 10
    well["exp287_sum_squared_error"] = 100.0
    well["exp264_sum_squared_error"] = 100.0
    well["exp287_rmse"] = 1.0
    well["exp264_rmse"] = 1.0
    well["delta_rmse_exp287_minus_exp264"] = 0.0
    well["verification_like_spatial_role"] = "valid"
    well["verification_like_typewell_purged_role"] = "valid"
    _, _, _, decision = exp336.evaluate_frozen_families(well, freeze, config)
    json.dumps(decision, allow_nan=False)
    plane = next(row for row in decision["families"] if row["family"] == "plane_reference_distance")
    assert plane["passed"] is False
    assert plane["global_q4_minus_q1_mean_well_delta_rmse"] is None


def test_fixed_decision_gate_requires_all_registered_checks(tmp_path: Path) -> None:
    config = config_for_synthetic_wells()
    attributes, freeze, _ = exp336.freeze_target_free_attributes(
        synthetic_target_free_attributes(),
        config=config,
        output_dir=tmp_path,
        partition_evidence=[],
        raw_context_audit={"forbidden_value_columns_opened": []},
    )
    well = attributes.copy()
    quartile = well["plane_reference_distance__risk_quartile"].to_numpy()
    well["rows"] = 10
    well["exp287_sum_squared_error"] = 100.0
    well["exp264_sum_squared_error"] = 100.0
    well["exp287_rmse"] = 1.0
    well["exp264_rmse"] = 1.0
    well["delta_rmse_exp287_minus_exp264"] = (quartile - 1) * 0.30
    well["verification_like_spatial_role"] = "valid"
    well["verification_like_typewell_purged_role"] = "valid"
    family_quartile, fold_metrics, hidden_metrics, decision = exp336.evaluate_frozen_families(
        well, freeze, config
    )
    assert len(family_quartile) == 24
    assert len(fold_metrics) == 30
    assert len(hidden_metrics) == 12
    plane = next(row for row in decision["families"] if row["family"] == "plane_reference_distance")
    assert plane["global_q4_minus_q1_mean_well_delta_rmse"] == pytest.approx(0.9)
    assert plane["positive_direction_folds"] == 5
    assert all(plane["checks"].values())
    assert plane["passed"] is True
    assert decision["status"] == "ATTRIBUTION_SUPPORTED"


def test_oof_alignment_and_well_rmse_endpoint() -> None:
    exp287_frame = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "well": ["w0", "w0", "w1", "w1"],
            "outer_fold": [0, 0, 1, 1],
            "actual_tvt": [0.0, 0.0, 0.0, 0.0],
            "pred287": [1.0, 1.0, 2.0, 2.0],
        }
    )
    exp264_frame = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "well": ["w0", "w0", "w1", "w1"],
            "outer_fold": [0, 0, 1, 1],
            "actual_tvt": [0.0, 0.0, 0.0, 0.0],
            "pred264": [0.5, 0.5, 1.0, 1.0],
        }
    )
    well, audit = exp336.build_well_oof_delta_metrics(
        exp287_frame,
        exp264_frame,
        exp287_prediction_column="pred287",
        exp264_prediction_column="pred264",
        actual_tolerance=1.0e-4,
    )
    assert audit["id_well_fold_alignment"] is True
    assert well.set_index("well").loc["w0", "delta_rmse_exp287_minus_exp264"] == pytest.approx(0.5)
    assert well.set_index("well").loc["w1", "delta_rmse_exp287_minus_exp264"] == pytest.approx(1.0)
