from __future__ import annotations

import importlib.util
import os
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
EXP399_DIR = ROOT / "experiments" / "exp399_soft_sticky_fused_exact_runtime_audit"
EXP394_DIR = ROOT / "experiments" / "exp394_soft_sticky_exp226_k16_branch_hmm"
EXP399_SOURCE = EXP399_DIR / (
    "exp399_soft_sticky_fused_exact_runtime_audit_compact_selfcontained_train.py"
)
EXP394_SOURCE = EXP394_DIR / (
    "exp394_soft_sticky_exp226_k16_branch_hmm_compact_selfcontained_train.py"
)


def load_source(name: str, path: Path, import_flag: str):
    previous = os.environ.get(import_flag)
    os.environ[import_flag] = "1"
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        if previous is None:
            os.environ.pop(import_flag, None)
        else:
            os.environ[import_flag] = previous


@pytest.fixture(scope="module")
def candidate():
    return load_source("exp399_train_test", EXP399_SOURCE, "EXP399_IMPORT_ONLY")


@pytest.fixture(scope="module")
def parent():
    return load_source("exp394_parent_for_exp399", EXP394_SOURCE, "EXP394_IMPORT_ONLY")


@pytest.fixture(scope="module")
def config():
    value = yaml.safe_load((EXP399_DIR / "config.yaml").read_text())
    assert isinstance(value, dict)
    return value


def test_contract_keeps_full_state_and_requires_separately_approved_full_oof(
    candidate, config
):
    ready = deepcopy(config)
    ready["implementation"]["compact_selfcontained_notebook_created"] = True
    ready["implementation"]["canonical_notebook_adopted"] = True
    counts = candidate.validate_scientific_contract(ready)
    assert counts["switching_hmm_well_runs"] == 773
    assert counts["boosters"] == 0
    assert ready["model"]["h_branch"]["retain_all_tvt_grid_states"] is True
    assert ready["model"]["h_branch"]["retain_all_rate_states"] is True
    assert (
        ready["validation"]["technical_preflight"]["parity"][
            "maximum_prediction_abs_diff_ft"
        ]
        == 1.0e-5
    )
    assert (
        ready["validation"]["technical_preflight"]["parent_reference"][
            "state_time_units"
        ]
        == 3_290_350_409
    )
    with pytest.raises(RuntimeError, match="run_technical_preflight is disabled"):
        candidate.validate_scientific_contract(ready, run_stage="technical_preflight")
    enabled = deepcopy(ready)
    enabled["execution"]["run_technical_preflight"] = True
    candidate.validate_scientific_contract(enabled, run_stage="technical_preflight")
    assert ready["execution"]["full_oof_approved"] is True
    full_enabled = deepcopy(ready)
    full_enabled["execution"]["run_full_oof"] = True
    candidate.validate_scientific_contract(full_enabled, run_stage="full_oof")
    unapproved = deepcopy(full_enabled)
    unapproved["execution"]["full_oof_approved"] = False
    with pytest.raises(RuntimeError, match="full OOF is not separately approved"):
        candidate.validate_scientific_contract(unapproved, run_stage="full_oof")


def test_sparse_r_by_5_position_kernel_matches_exp394_boundaries(candidate, parent):
    grid = np.arange(17, dtype=np.float64) * 0.35 + 1000.0
    rates = np.linspace(-0.1, 0.1, 7)
    delta_md = 1.4
    effective_delta_z = 0.19
    parent_destination, parent_logp, parent_error = parent._position_kernel(
        grid,
        rates,
        delta_md,
        effective_delta_z,
        0.02,
    )
    offsets, probability, logp, candidate_error = (
        candidate._exact_sparse_position_kernel(
            rates,
            delta_md,
            effective_delta_z,
            0.02,
            0.35,
            len(grid),
        )
    )
    reconstructed_destination = np.full_like(parent_destination, -1)
    reconstructed_logp = np.full_like(parent_logp, -np.inf)
    for destination_rate in range(len(rates)):
        for source_position in range(len(grid)):
            valid_mass = candidate._source_boundary_mass(
                source_position,
                destination_rate,
                offsets,
                probability,
                len(grid),
            )
            for slot in range(5):
                destination_position = (
                    source_position + offsets[destination_rate] - 2 + slot
                )
                if 0 <= destination_position < len(grid):
                    reconstructed_destination[
                        destination_rate, source_position, slot
                    ] = destination_position
                    reconstructed_logp[destination_rate, source_position, slot] = (
                        logp[destination_rate, slot] - np.log(valid_mass)
                    )
    np.testing.assert_array_equal(reconstructed_destination, parent_destination)
    np.testing.assert_allclose(reconstructed_logp, parent_logp, atol=2.0e-14)
    assert parent_error <= 1.0e-14
    assert candidate_error <= 1.0e-14


def test_fused_exact_forward_backward_matches_exp394(candidate, parent):
    row_count, position_count, rate_count = 9, 17, 7
    grid = np.arange(position_count, dtype=np.float64) * 0.35 + 1000.0
    rates = np.linspace(-0.03, 0.03, rate_count)
    delta_md = np.array([0.7, 0.8, 1.1, 0.9, 1.2, 0.75, 1.05, 0.95, 0.85])
    effective_delta_z = np.array(
        [0.01, -0.02, 0.03, 0.0, -0.01, 0.02, -0.03, 0.01, 0.0]
    )
    q_e = np.linspace(1001.0, 1004.0, row_count)
    h_emission = -0.5 * ((grid[None, :] - q_e[:, None]) / 1.4) ** 2
    e_emission = np.linspace(-1.5, -0.5, row_count)
    arguments = (
        h_emission,
        e_emission,
        grid,
        rates,
        delta_md,
        effective_delta_z,
        q_e,
        0.35,
        0.002,
        0.02,
        4.0,
        0.75,
        0.0,
        0.01,
        0.998,
        1000.0,
        6.0,
        0.5,
    )
    expected = parent._soft_sticky_forward_backward(*arguments)
    actual = candidate._soft_sticky_forward_backward(*arguments)
    for index, (expected_value, actual_value) in enumerate(
        zip(expected, actual, strict=True)
    ):
        tolerance = 2.0e-6 if index in {2, 3, 4, 5} else 2.0e-8
        np.testing.assert_allclose(
            np.asarray(actual_value),
            np.asarray(expected_value),
            atol=tolerance,
            rtol=0.0,
        )


def test_source_uses_fused_sparse_kernel_and_worker_local_thread_mask():
    source = EXP399_SOURCE.read_text()
    core = source[source.index("def _soft_sticky_forward_backward") :]
    assert "_exact_sparse_position_kernel(" in core
    assert "_fused_separable_forward_transition(" in core
    assert "_fused_backward_rate_switch(" in core
    assert ") = _position_kernel(" not in core
    assert "set_num_threads(int(get_nested(config, \"runtime.numba_num_threads\")))" in source
    assert 'Parallel(n_jobs=outer_workers, prefer="threads")' in source
    assert "from exp394_" not in source
    assert "import exp394_" not in source


def test_parent_key_parity_allows_csv_integer_dtype_widening(candidate):
    left = pd.DataFrame(
        {
            "well_id": pd.Series(["a", "b"], dtype="str"),
            "row_idx": np.array([10, 20], dtype=np.int32),
        }
    )
    right = pd.DataFrame(
        {
            "well_id": pd.Series(["a", "b"], dtype="object"),
            "row_idx": np.array([10, 20], dtype=np.int64),
        }
    )
    assert candidate.same_key_values(left, right)
    right.loc[1, "row_idx"] = 21
    assert not candidate.same_key_values(left, right)


def test_late_readout_accepts_independent_grouped_fold_ledgers(candidate):
    frame = pd.DataFrame(
        {
            "well_id": ["a", "a", "b", "b", "c", "c", "d", "d", "e", "e"],
            "fold": np.repeat(np.arange(5, dtype=np.int8), 2),
            "exp263_fold": np.repeat(
                np.array([1, 2, 3, 4, 0], dtype=np.int8),
                2,
            ),
        }
    )
    report = candidate.audit_independent_fold_assignments(frame)
    assert report["row_identity_match"] is False
    assert report["mismatch_rows"] == len(frame)
    assert report["mismatch_wells"] == 5
    assert (
        report["relationship"]
        == "independent_grouped_oof_ledgers_not_required_to_match"
    )

    invalid = pd.concat(
        [
            frame,
            pd.DataFrame(
                {
                    "well_id": ["a"],
                    "fold": [1],
                    "exp263_fold": [1],
                }
            ),
        ],
        ignore_index=True,
    )
    with pytest.raises(RuntimeError, match="fold is not constant within well"):
        candidate.audit_independent_fold_assignments(invalid)


def test_full_oof_checkpoints_frozen_predictions_before_late_readout():
    source = EXP399_SOURCE.read_text()
    freeze_index = source.index("frozen = freeze_predictions(")
    oof_checkpoint_index = source.index(
        'artifacts / f"{prefix}_oof_predictions.csv.gz"',
        freeze_index,
    )
    late_readout_index = source.index(
        "frame, late_report = attach_late_readout(",
        freeze_index,
    )
    assert freeze_index < oof_checkpoint_index < late_readout_index
