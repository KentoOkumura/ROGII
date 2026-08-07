from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = (
    ROOT
    / "experiments"
    / "exp490_geometry_centered_mean_reverting_offset_hmm"
    / "exp490_geometry_centered_mean_reverting_offset_hmm_compact_selfcontained_train.py"
)


def load_module():
    name = "exp490_compact_selfcontained_train_contract"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_k16_matches_exp226_equal_row_segmentation_and_half_life() -> None:
    module = load_module()
    dmd = np.linspace(0.75, 1.25, 32)
    md = 1000.0 + np.cumsum(dmd)
    contract = module.k16_segment_half_life(
        md,
        last_known_md=1000.0,
        segment_count=16,
    )
    edges = np.linspace(0.0, 32.0, 17)
    expected_segment = np.searchsorted(
        edges[1:],
        np.arange(1, 33, dtype=float),
        side="left",
    )
    np.testing.assert_array_equal(contract["segment_id"], expected_segment)
    np.testing.assert_allclose(
        contract["segment_cumulative_rho"],
        np.full(16, 0.5),
        rtol=0.0,
        atol=1.0e-12,
    )


def test_destination_row_owns_boundary_transition_span() -> None:
    module = load_module()
    dmd = np.ones(32)
    dmd[2] = 7.0
    md = 500.0 + np.cumsum(dmd)
    contract = module.k16_segment_half_life(
        md,
        last_known_md=500.0,
        segment_count=16,
    )
    assert contract["segment_id"][1] == 0
    assert contract["segment_id"][2] == 1
    assert contract["segment_span"][0] == pytest.approx(2.0)
    assert contract["segment_span"][1] == pytest.approx(8.0)
    assert contract["rho"][2] == pytest.approx(2.0 ** (-7.0 / 8.0))


def test_nonpositive_dmd_fails_closed() -> None:
    module = load_module()
    md = 100.0 + np.arange(32, dtype=float)
    md[8] = md[7]
    with pytest.raises(ValueError, match="strictly positive"):
        module.k16_segment_half_life(
            md,
            last_known_md=99.0,
            segment_count=16,
        )


def test_zero_state_is_exact_geometry_identity() -> None:
    module = load_module()
    dmd = np.asarray([1.0, 2.0, 3.0])
    rho = np.asarray([0.99, 0.95, 0.90])
    sentinel = module.zero_state_geometry_identity(dmd, rho)
    assert sentinel["pass"]
    assert sentinel["maximum_abs_offset_ft"] == 0.0


def test_truth_role_fold_episode_reads_are_blocked_until_full_freeze() -> None:
    module = load_module()
    ledger = module.LeakageLedger(expected_wells=1)
    with pytest.raises(RuntimeError, match="before candidate/contract freeze"):
        ledger.record_forbidden("truth_role_fold_episode", 3)
    assert ledger.forbidden_reads_before_freeze == {
        "truth_role_fold_episode": 3
    }

    frozen = module.LeakageLedger(expected_wells=1)
    frozen.freeze_prediction("well-a", "prediction-sha")
    frozen.freeze_decoder_contract("decoder-contract-sha")
    frozen.record_forbidden("truth_role_fold_episode", 3)
    assert frozen.forbidden_reads_before_freeze == {}
    assert frozen.post_freeze_reads == {"truth_role_fold_episode": 3}


def test_scaled_posterior_is_finite_and_normalized() -> None:
    module = load_module()
    position_count = len(np.arange(-80.0, 80.0 + 0.5 * 0.35, 0.35))
    emission = np.zeros((2, position_count), dtype=np.float32)
    rates = np.linspace(-0.10, 0.10, 41)
    posterior, log_likelihood, maximum_error = (
        module._hmm2_fb_geometry_mean_reversion(
            emission,
            np.ones(2, dtype=np.float64),
            np.full(2, 0.999, dtype=np.float64),
            0.35,
            rates,
            0.002,
            0.02,
            80.0 / 0.35,
            0.75,
            0.0,
            0.01,
            1.0,
            0.998,
        )
    )
    assert np.isfinite(log_likelihood)
    assert np.isfinite(posterior).all()
    np.testing.assert_allclose(
        posterior.sum(axis=1),
        np.ones(2),
        rtol=0.0,
        atol=1.0e-6,
    )
    assert maximum_error <= 1.0e-6
