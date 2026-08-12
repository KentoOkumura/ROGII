from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = (
    ROOT
    / "experiments"
    / "exp491_exp226_final_tvt_rate_direct_hmm"
)
SOURCE = (
    EXP_DIR
    / "exp491_exp226_final_tvt_rate_direct_hmm_compact_selfcontained_train.py"
)


def load_module():
    name = "exp491_compact_selfcontained_train_contract"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_config() -> dict:
    return yaml.safe_load((EXP_DIR / "config.yaml").read_text())


def test_implementation_contract_is_single_candidate_stage0_only() -> None:
    module = load_module()
    config = load_config()
    counts = module.validate_execution_contract(
        config,
        require_run_authorization=False,
    )
    assert counts["current_scientific_variants"] == 1
    assert counts["current_hmm_well_runs"] == 32
    assert counts["parent_control_hmm_reruns"] == 0
    assert counts["boosters"] == 0
    assert counts["pf_runs"] == 0
    assert config["design"]["kaggle_stage_0_authorized"]
    assert not config["runtime"]["run_approved"]
    assert not config["execution"]["run_hmm"]
    assert not config["design"]["kaggle_stage_1_authorized"]
    assert not config["design"]["inference_authorized"]
    assert not config["design"]["submission_authorized"]
    with pytest.raises(RuntimeError, match="runtime.run_approved is false"):
        module.validate_execution_contract(
            config,
            require_run_authorization=True,
        )


def test_scientific_contract_keeps_exp437_hmm_and_final_tvt_only() -> None:
    module = load_module()
    contract = module.validate_scientific_contract(load_config())
    assert contract["variant"] == "exp226_final_rate_direct_transition"
    assert contract["parent"] == "exp437_neighbor_geometry_tvt_only_transition_hmm"
    assert contract["persistent_state"] == "tvt_probability_distribution_only"
    assert contract["rate_state_present"] is False
    assert contract["residual_offset_state_present"] is False
    assert contract["branch_state_present"] is False
    assert contract["parent_control_hmm_reruns"] == 0
    assert contract["exp226_allowlist"] == [
        "well_id",
        "row_idx",
        "suffix_offset",
        "fold",
        "tvt_pred",
    ]
    assert "tvt_geop" in contract["exp226_forbidden"]


def test_final_tvt_schedule_first_difference_and_rate_identity() -> None:
    module = load_module()
    source = pd.DataFrame(
        {
            "well_id": ["a", "a", "a"],
            "row_idx": [10, 11, 12],
            "suffix_offset": [0, 1, 2],
            "fold": [2, 2, 2],
            "tvt_pred": [101.5, 102.0, 103.25],
        }
    )
    schedule = module.build_final_tvt_schedule(
        source,
        expected_row_idx=np.asarray([10, 11, 12]),
        suffix_md=np.asarray([201.0, 203.0, 206.0]),
        suffix_z=np.asarray([51.0, 50.5, 49.0]),
        last_known_tvt=100.0,
        last_known_md=200.0,
        last_known_z=50.0,
    )
    np.testing.assert_allclose(
        schedule["transition_delta"],
        np.asarray([1.5, 0.5, 1.25]),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        schedule["delta_md"],
        np.asarray([1.0, 2.0, 3.0]),
        rtol=0.0,
        atol=0.0,
    )
    assert schedule["first_difference_parity_max_abs_ft"] == 0.0
    assert schedule["rate_increment_identity_max_abs_ft"] <= 1.0e-15


def test_nonpositive_delta_md_fails_closed() -> None:
    module = load_module()
    source = pd.DataFrame(
        {
            "well_id": ["a", "a"],
            "row_idx": [10, 11],
            "suffix_offset": [0, 1],
            "fold": [0, 0],
            "tvt_pred": [101.0, 102.0],
        }
    )
    with pytest.raises(ValueError, match="strictly positive"):
        module.build_final_tvt_schedule(
            source,
            expected_row_idx=np.asarray([10, 11]),
            suffix_md=np.asarray([201.0, 201.0]),
            suffix_z=np.asarray([50.0, 50.0]),
            last_known_tvt=100.0,
            last_known_md=200.0,
            last_known_z=50.0,
        )


def test_truth_role_and_episode_reads_are_blocked_until_full_freeze() -> None:
    module = load_module()
    for method_name in ("record_truth", "record_roles", "record_episodes"):
        ledger = module.LeakageLedger(expected_wells=1)
        with pytest.raises(RuntimeError, match="before all candidates froze"):
            getattr(ledger, method_name)(3)

    ledger = module.LeakageLedger(expected_wells=1)
    ledger.freeze("well-a")
    ledger.record_truth(3)
    ledger.record_roles(1)
    ledger.record_episodes(1)
    assert ledger.truth_rows_after_all_freeze == 3
    assert ledger.role_rows_after_all_freeze == 1
    assert ledger.episode_rows_after_all_freeze == 1


def test_combined_prediction_artifact_freezes_before_forbidden_readout() -> None:
    source = SOURCE.read_text()
    run_stage0 = source[source.index("def run_stage0(") :]
    prediction_write = run_stage0.index(
        "prediction_artifact = write_deterministic_gzip_csv("
    )
    role_read = run_stage0.index("scope_manifest = load_scope_after_freeze(")
    episode_read = run_stage0.index(
        "episodes, episode_input = load_persistent_episodes_after_freeze("
    )
    truth_read = run_stage0.index(
        "readout = truth_late_frame(frozen_wells, raw_dir, ledger)"
    )
    assert prediction_write < role_read < episode_read < truth_read


def test_deterministic_gzip_writer_closes_stream_before_readback(
    tmp_path: Path,
) -> None:
    module = load_module()
    frame = module.pd.DataFrame(
        {
            "well": ["a", "b"],
            "prediction": [1.25, 2.5],
        }
    )
    output = tmp_path / "predictions.csv.gz"

    artifact = module.write_deterministic_gzip_csv(output, frame)
    readback = module.pd.read_csv(output, float_precision="round_trip")

    module.pd.testing.assert_frame_equal(readback, frame)
    assert artifact["rows"] == 2
    assert artifact["logical_sha256"] == artifact["readback_logical_sha256"]


def test_position_kernel_is_normalized_and_fixed_to_five_cells() -> None:
    module = load_module()
    offsets, probabilities, error = module.direct_position_kernel(
        expected_delta=1.37,
        step=0.35,
        sig_p=0.02,
    )
    assert len(offsets) == 5
    assert len(probabilities) == 5
    np.testing.assert_allclose(
        probabilities.sum(),
        1.0,
        rtol=0.0,
        atol=1.0e-12,
    )
    assert error <= 1.0e-12


def test_direct_hmm_posterior_is_finite_and_normalized() -> None:
    module = load_module()
    emission = np.zeros((4, 81), dtype=np.float32)
    posterior, log_likelihood, transition_error, forward_error, posterior_error = (
        module._direct_transition_forward_backward(
            emission,
            np.asarray([0.0, 0.35, -0.35, 0.7], dtype=np.float64),
            0.35,
            0.02,
            40.0,
            0.75,
            1.0,
        )
    )
    assert np.isfinite(log_likelihood)
    assert np.isfinite(posterior).all()
    np.testing.assert_allclose(
        posterior.sum(axis=1),
        np.ones(4),
        rtol=0.0,
        atol=1.0e-6,
    )
    assert transition_error <= 1.0e-6
    assert forward_error <= 1.0e-6
    assert posterior_error <= 1.0e-6
