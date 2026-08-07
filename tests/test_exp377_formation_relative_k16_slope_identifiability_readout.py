from __future__ import annotations

import os
import runpy
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

EXPERIMENT_DIR = Path(
    "experiments/exp377_formation_relative_k16_slope_identifiability_readout"
)
SOURCE_PATH = EXPERIMENT_DIR / (
    "exp377_formation_relative_k16_slope_identifiability_readout_"
    "compact_selfcontained_train.py"
)
CONFIG_PATH = EXPERIMENT_DIR / "config.yaml"


def load_namespace() -> dict[str, object]:
    previous = os.environ.get("EXP377_IMPORT_ONLY")
    os.environ["EXP377_IMPORT_ONLY"] = "1"
    try:
        return runpy.run_path(str(SOURCE_PATH))
    finally:
        if previous is None:
            os.environ.pop("EXP377_IMPORT_ONLY", None)
        else:
            os.environ["EXP377_IMPORT_ONLY"] = previous


@pytest.fixture(scope="module")
def module() -> dict[str, object]:
    return load_namespace()


def test_execution_contract_is_zero_model_hmm_pf_and_v2_run_is_completed(
    module: dict[str, object],
) -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text())
    module["validate_execution_contract"](
        config, require_kaggle_authorization=True
    )
    assert config["execution"]["implementation_authorized"] is True
    assert config["execution"]["kaggle_execution_authorized"] is True
    assert config["execution"]["canonical_train_notebook_adopted"] is True
    assert config["execution"]["one_run_authorization_consumed"] is True
    assert config["execution"]["kaggle_v1_execution_completed"] is True
    assert config["execution"]["kaggle_v2_execution_authorized"] is True
    assert config["execution"]["kaggle_v2_run_authorization_consumed"] is True
    assert config["execution"]["kaggle_v2_execution_started"] is True
    assert config["execution"]["kaggle_execution_completed"] is True
    assert config["execution"]["kaggle_execution_success"] is True
    assert config["execution"]["kaggle_v2_execution_completed"] is True
    assert config["execution"]["kaggle_v2_execution_success"] is True
    assert config["execution"]["stage0_passed"] is True
    assert config["execution"]["stage1_passed"] is False
    assert config["execution"]["scientific_support"] is False
    assert config["execution"]["kaggle_push_enabled"] is False
    assert config["execution"]["kaggle_run_enabled"] is False
    assert config["execution"]["inference_enabled"] is False
    assert config["execution"]["submission_enabled"] is False
    assert config["runtime"]["scientific_variants"] == 1
    assert config["runtime"]["reporting_surfaces"] == 6
    assert config["runtime"]["reporting_folds"] == 5
    assert config["runtime"]["model_configs"] == 0
    assert config["runtime"]["lightgbm_boosters"] == 0
    assert config["runtime"]["hmm_runs"] == 0
    assert config["runtime"]["pf_runs"] == 0
    assert config["runtime"]["kaggle"]["enable_gpu"] is False
    assert config["runtime"]["kaggle"]["enable_internet"] is False
    assert config["runtime"]["kaggle"]["train_run_on_push"] is False
    assert config["gates"]["stage0_integrity"]["report_only_checks"] == [
        "effective_donors_p05"
    ]


def test_k16_segmentation_and_median_step_rate_contract(
    module: dict[str, object],
) -> None:
    segment_id = module["k16_segment_ids"](32, 16)
    assert np.array_equal(np.bincount(segment_id), np.full(16, 2))
    md = np.arange(33, dtype=float)
    values = 2.5 * md + 7.0
    suffix_rows = np.arange(1, 33, dtype=np.int64)
    rates, counts = module["median_segment_step_rates"](
        md,
        values,
        suffix_rows,
        segment_id,
        16,
    )
    assert np.allclose(rates, 2.5)
    assert np.array_equal(counts, np.ones(16, dtype=np.int32))


def test_formation_plane_is_exact_for_planar_surfaces_and_excludes_self(
    module: dict[str, object],
) -> None:
    grid_x, grid_y = np.meshgrid(np.arange(4.0), np.arange(3.0))
    xy = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    wells = np.asarray([f"w{index:02d}" for index in range(len(xy))])
    surfaces = np.column_stack(
        [
            100.0 * column + 2.0 * xy[:, 0] - 3.0 * xy[:, 1]
            for column in range(6)
        ]
    )
    plane = module["FormationPlaneKNN"](
        wells=wells,
        xy=xy,
        formation_medians=surfaces,
        k=10,
    )
    query = np.asarray([[1.25, 0.75]])
    prediction, support = plane.predict(query)
    expected = np.asarray(
        [100.0 * column + 2.0 * 1.25 - 3.0 * 0.75 for column in range(6)]
    )
    assert np.allclose(prediction[0], expected, atol=1.0e-7)
    assert not support["fallback"][0]
    assert support["effective_donors"][0] > 1.0

    contaminated = surfaces.copy()
    contaminated[0] += 100_000.0
    self_plane = module["FormationPlaneKNN"](
        wells=wells,
        xy=xy,
        formation_medians=contaminated,
        k=10,
    )
    with_self, _ = self_plane.predict(xy[:1])
    without_self, _ = self_plane.predict(xy[:1], target_well="w00")
    assert float(with_self[0, 0]) > float(without_self[0, 0]) + 1_000.0


def test_relative_rate_reconstruction_and_preregistered_median(
    module: dict[str, object],
) -> None:
    # True dS/dMD=2.0. A formation surface contributes dF/dMD=0.5,
    # so the projected relative component is 1.5 at unit projection.
    reconstructed = module["reconstruct_relative_rate"](
        np.asarray([1.5, 1.5]),
        np.asarray([1.0, 0.5]),
        np.asarray([0.5, 1.25]),
    )
    assert np.allclose(reconstructed, 2.0)

    six = np.column_stack(
        [
            reconstructed,
            reconstructed + 0.1,
            reconstructed - 0.1,
            reconstructed + 0.2,
            reconstructed - 0.2,
            reconstructed + 20.0,
        ]
    )
    primary, finite_count = module["robust_median_primary"](six, 6)
    assert np.array_equal(finite_count, np.full(2, 6))
    assert np.allclose(primary, 2.05)
    six[0, 0] = np.nan
    strict_primary, strict_count = module["robust_median_primary"](six, 6)
    assert strict_count[0] == 5
    assert np.isnan(strict_primary[0])


def test_xy_kernel_uses_stable_k_and_reports_effective_support(
    module: dict[str, object],
) -> None:
    rows = []
    for well_index in range(20):
        for segment in range(3):
            x = float(well_index * 10 + segment)
            y = float(segment * 4)
            rows.append(
                {
                    "x": x,
                    "y": y,
                    "value": 3.0 + 0.002 * x - 0.001 * y,
                    "donor_well": f"w{well_index:02d}",
                    "donor_segment": segment,
                }
            )
    field = pd.DataFrame(rows)
    contract = module["K16Contract"](
        local_linear_k=50,
        bandwidth_ft=500.0,
        ridge=1.0,
    )
    first = module["local_linear_predict"](
        field,
        np.asarray([[50.0, 2.0], [75.0, 6.0]]),
        contract,
    )
    second = module["local_linear_predict"](
        field.sample(frac=1.0, random_state=9).reset_index(drop=True),
        np.asarray([[50.0, 2.0], [75.0, 6.0]]),
        contract,
    )
    assert np.allclose(first.values, second.values)
    assert np.all(first.selected_donors == 50)
    assert np.all(first.effective_donors >= 49.0)
    assert not first.fallback.any()


def test_segment_rate_integration_is_anchored_in_structural_position(
    module: dict[str, object],
) -> None:
    rows = 33
    md = np.arange(rows, dtype=float)
    z = -md
    tvt_input = np.full(rows, np.nan)
    tvt_input[0] = 10.0
    suffix_row_idx = np.arange(1, rows, dtype=np.int64)
    segment_id = module["k16_segment_ids"](len(suffix_row_idx), 16)
    start, end = module["segment_bounds"](suffix_row_idx, segment_id, 16)
    geometry = module["WellGeometry"](
        well_id="w0",
        fold=0,
        row_count=rows,
        anchor_row_idx=0,
        suffix_row_idx=suffix_row_idx,
        suffix_offset=np.arange(32, dtype=np.int32),
        segment_id=segment_id,
        segment_start_idx=start,
        segment_end_idx=end,
        segment_mid_xy=np.zeros((16, 2)),
        segment_projection=np.ones(16),
        x=np.zeros(rows),
        y=np.zeros(rows),
        z=z,
        md=md,
        tvt_input=tvt_input,
    )
    prediction = module["integrate_segment_rates"](
        geometry,
        np.full(16, 2.0),
    )
    # S(anchor)=10 and dS/dMD=2, while Z=-MD, hence TVT=10+3*MD.
    assert np.allclose(prediction, 10.0 + 3.0 * md[1:])


def test_stage1_identifiability_is_a_fixed_and_gate(
    module: dict[str, object],
) -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text())
    segment_records = [
        {
            "candidate": module["PRIMARY_CANDIDATE"],
            "scope": "pooled",
            "scope_value": "all",
            "baseline_rmse": 10.0,
            "candidate_rmse": 9.0,
            "gain_baseline_minus_candidate": 1.0,
            "gain_fraction": 0.10,
        }
    ]
    path_records = [
        {
            "candidate": module["PRIMARY_CANDIDATE"],
            "scope": "pooled",
            "scope_value": "all",
            "baseline_rmse": 10.0,
            "candidate_rmse": 9.0,
            "gain_baseline_minus_candidate": 1.0,
            "gain_fraction": 0.10,
        }
    ]
    for fold in range(5):
        segment_records.append(
            {
                "candidate": module["PRIMARY_CANDIDATE"],
                "scope": "fold",
                "scope_value": str(fold),
                "baseline_rmse": 10.0,
                "candidate_rmse": 9.0,
                "gain_baseline_minus_candidate": 1.0,
                "gain_fraction": 0.10,
            }
        )
        path_records.append(
            {
                "candidate": module["PRIMARY_CANDIDATE"],
                "scope": "fold",
                "scope_value": str(fold),
                "baseline_rmse": 10.0,
                "candidate_rmse": 9.0,
                "gain_baseline_minus_candidate": 1.0,
                "gain_fraction": 0.10,
            }
        )
    for scope in config["validation"]["scopes"]:
        path_records.append(
            {
                "candidate": module["PRIMARY_CANDIDATE"],
                "scope": "scope",
                "scope_value": scope,
                "baseline_rmse": 10.0,
                "candidate_rmse": 9.99,
                "delta_candidate_minus_baseline": -0.01,
                "gain_baseline_minus_candidate": 0.01,
                "gain_fraction": 0.001,
            }
        )
    by_well = pd.DataFrame(
        {
            "well_id": ["a", "b", "c"],
            "delta_candidate_minus_baseline": [-1.0, -0.5, -0.1],
        }
    )
    passing = module["evaluate_stage1_identifiability"](
        pd.DataFrame(segment_records),
        pd.DataFrame(path_records),
        by_well,
        config,
    )
    assert passing["passed"] is True
    by_well.loc[2, "delta_candidate_minus_baseline"] = 0.30
    failing = module["evaluate_stage1_identifiability"](
        pd.DataFrame(segment_records),
        pd.DataFrame(path_records),
        by_well,
        config,
    )
    assert failing["passed"] is False
    assert failing["checks"]["worst_well_guard"] is False


def make_synthetic_stage0_bundle(
    module: dict[str, object],
    *,
    effective_donors: float,
    valid_formation_reads: int,
) -> object:
    primary = module["PRIMARY_CANDIDATE"]
    paths = pd.DataFrame(
        {
            "well_id": ["w0", "w0"],
            "row_idx": [1, 2],
            "suffix_offset": [0, 1],
            "suffix_rows": [2, 2],
            "fold": [0, 0],
            "segment_id": [0, 1],
            "md_since": [1.0, 2.0],
            "baseline_direct_path": [1.0, 2.0],
            primary: [1.0, 2.0],
        }
    )
    segment = pd.DataFrame(
        {
            "well_id": ["w0"] * 16,
            "segment_id": np.arange(16),
            "fold": [0] * 16,
            primary: np.ones(16),
            "surface_fallback": [False] * 16,
            "effective_donors_min": np.full(16, effective_donors),
        }
    )
    role = pd.DataFrame(
        {
            "outer_fold": [0, 0],
            "role": ["outer_train_source", "outer_valid_target"],
            "wells": [1, 1],
            "truth_file_reads": [1, 0],
            "formation_file_reads": [1, valid_formation_reads],
        }
    )
    return module["TargetFreeBundle"](
        input_manifest=pd.DataFrame(),
        fold_manifest=pd.DataFrame(
            {"outer_fold": [0], "source_valid_overlap": [0]}
        ),
        role_read_ledger=role,
        donor_fields=pd.DataFrame(),
        segment_schedule=segment,
        primary_paths=paths,
        artifact_evidence={
            key: {}
            for key in (
                "input_manifest",
                "fold_manifest",
                "role_read_ledger",
                "donor_fields",
                "segment_schedule",
                "primary_paths",
            )
        },
        freeze_manifest={
            "truth_access_before_freeze": 0,
            "bundle_logical_sha256": "synthetic",
        },
    )


def synthetic_stage0_config() -> dict[str, object]:
    config = yaml.safe_load(CONFIG_PATH.read_text())
    config = deepcopy(config)
    config["gates"]["stage0_integrity"].update(
        {
            "expected_rows": 2,
            "expected_wells": 1,
            "expected_segments": 16,
            "expected_outer_fold_runs": 1,
        }
    )
    config["validation"]["expected_folds"] = [0]
    return config


def test_stage0_support_is_report_only_for_shared_exp226_kernel(
    module: dict[str, object],
) -> None:
    config = synthetic_stage0_config()
    bundle = make_synthetic_stage0_bundle(
        module,
        effective_donors=2.5,
        valid_formation_reads=0,
    )
    result = module["evaluate_stage0_integrity"](bundle, config)
    assert result["passed"] is True
    assert result["checks"]["effective_donors_p05"] is False
    assert result["report_only_checks"]["effective_donors_p05"] is False
    assert result["report_only_warning"] is True
    assert result["fail_action"] is None


def test_stage0_guard_fails_closed_on_valid_role_formation_read(
    module: dict[str, object],
) -> None:
    config = synthetic_stage0_config()
    bundle = make_synthetic_stage0_bundle(
        module,
        effective_donors=20.0,
        valid_formation_reads=1,
    )
    result = module["evaluate_stage0_integrity"](bundle, config)
    assert result["passed"] is False
    assert result["checks"]["validation_truth_reads_zero"] is True
    assert result["checks"]["validation_formation_reads_zero"] is False
    assert result["fail_action"].startswith("stop_before_truth_join")
