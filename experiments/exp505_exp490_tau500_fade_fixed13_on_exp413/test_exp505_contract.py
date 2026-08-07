from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
SOURCE = (
    HERE
    / "exp505_exp490_tau500_fade_fixed13_on_exp413_compact_selfcontained_train.py"
)


def load_module():
    root = HERE.parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location("exp505_train", SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frozen_candidate_and_compute_contract() -> None:
    module = load_module()
    module.validate_exp505_contract(module.CONTRACT)
    assert module.candidate_order[-1] == module.ADDED_CANDIDATE_ID
    assert module.RAW_EXP490_CANDIDATE_ID not in module.candidate_order
    assert len(module.candidate_order) == 13
    assert len(module.compact_names) == 77
    assert module.cost_contract == {
        "variants": 1,
        "objectives": 2,
        "outer_folds": 5,
        "inner_folds": 4,
        "planned_cpu_selector_boosters": 40,
        "parent_control_retraining_boosters": 0,
        "new_hmm_well_runs": 0,
        "new_pf_well_runs": 0,
        "new_beam_well_runs": 0,
        "stage_d_enabled": False,
        "inference_runs": 0,
        "submission_runs": 0,
    }
    assert module.stage_c_execution["run_approved"] is True
    assert module.stage_c_execution["enabled"] is False
    assert module.stage_c_execution["run_completed"] is True
    assert module.stage_c_execution["rerun_approved"] is False
    assert module.stage_c_execution["approved_scope"] == (
        "stage_c_tau500_fade_fixed13_1_variant_2_objectives_"
        "5_outer_4_inner_40_cpu_boosters_no_control_retraining"
    )
    assert module.CONFIG["authorization"]["stage_d_implementation_approved"] is False


def test_fade_formula_is_exactly_alpha1_tau500() -> None:
    module = load_module()
    md = np.array([0.0, 500.0, 1000.0])
    parent = np.array([10.0, 10.0, 10.0])
    candidate = np.array([20.0, 20.0, 20.0])
    expected = parent + (1.0 - np.exp(-md / 500.0)) * (candidate - parent)
    assert np.allclose(module.fade_prediction(md, parent, candidate), expected)
    assert module.fade_prediction(md[:1], parent[:1], candidate[:1])[0] == parent[0]


def test_fade_rejects_grid_or_negative_distance() -> None:
    module = load_module()
    with np.testing.assert_raises(ValueError):
        module.fade_prediction(np.array([1.0]), np.array([0.0]), np.array([1.0]), tau_ft=85.0)
    with np.testing.assert_raises(ValueError):
        module.fade_prediction(np.array([-1.0]), np.array([0.0]), np.array([1.0]))


def test_loader_reads_exact_eight_column_allowlist(tmp_path: Path) -> None:
    module = load_module()
    source = pd.DataFrame(
        {
            "well": ["a", "a", "b"],
            "row_idx": [0, 1, 0],
            "suffix_offset": [0, 1, 0],
            "md_since": [0.0, 500.0, 100.0],
            "geometry_mean_reverting_hmm": [20.0, 20.0, 30.0],
            "exp357_parent_prediction": [10.0, 10.0, 20.0],
            "geometry_mean_reverting_delta_mean": [1.0, 1.5, 2.0],
            "geometry_mean_reverting_hmm_std": [2.0, 2.5, 3.0],
            "fold": [0, 0, 1],
            "true_tvt_readout_only": [11.0, 12.0, 21.0],
        }
    )
    path = tmp_path / "prediction.csv.gz"
    source.to_csv(path, index=False, compression="gzip")
    frame, manifest = module.load_exp490_fade_inputs(
        path,
        expected_rows=3,
        expected_wells=2,
        expected_prediction_gzip_raw_sha256=module.sha256_file(path),
        expected_prediction_payload_sha256=module.sha256_csv_payload(path),
        alpha=1.0,
        tau_ft=500.0,
    )
    assert manifest["prediction_loaded_columns"] == list(module.EXP490_INPUT_ALLOWLIST)
    assert manifest["loaded_column_count"] == 8
    assert manifest["forbidden_truth_error_role_episode_fold_scope_gate_columns_loaded"] == 0
    assert manifest["fade_formula_recompute_max_abs_ft"] == 0.0
    assert "fold" not in frame.columns
    assert "true_tvt_readout_only" not in frame.columns
    expected = module.fade_prediction(
        source["md_since"].to_numpy(),
        source["exp357_parent_prediction"].to_numpy(),
        source["geometry_mean_reverting_hmm"].to_numpy(),
    )
    assert np.allclose(frame["candidate_tvt"], expected)


def test_source_contains_no_forbidden_execution_path() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    forbidden_calls = (
        "run_particle_filter(",
        "run_pf(",
        "run_beam(",
        "stage_d_downstream(",
        "submission.csv",
    )
    assert all(call not in source for call in forbidden_calls)
