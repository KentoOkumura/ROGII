from __future__ import annotations

import importlib.util
import itertools
import os
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = ROOT / "experiments" / "exp394_soft_sticky_exp226_k16_branch_hmm"
TRAIN_SOURCE = EXP_DIR / ("exp394_soft_sticky_exp226_k16_branch_hmm_compact_selfcontained_train.py")


def load_module(name: str = "exp394_train_test"):
    previous = os.environ.get("EXP394_IMPORT_ONLY")
    os.environ["EXP394_IMPORT_ONLY"] = "1"
    try:
        spec = importlib.util.spec_from_file_location(name, TRAIN_SOURCE)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        if previous is None:
            os.environ.pop("EXP394_IMPORT_ONLY", None)
        else:
            os.environ["EXP394_IMPORT_ONLY"] = previous


@pytest.fixture(scope="module")
def module():
    return load_module()


@pytest.fixture(scope="module")
def config():
    value = yaml.safe_load((EXP_DIR / "config.yaml").read_text())
    assert isinstance(value, dict)
    return value


def test_frozen_contract_has_one_cpu_variant_and_completed_preflight_fails_closed(
    module,
    config,
):
    assert module.validate_scientific_contract(config) == {
        "scientific_variants": 1,
        "reporting_folds": 5,
        "switching_hmm_well_runs": 773,
        "lightgbm_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
        "parent_control_reruns": 0,
        "gpu": False,
    }
    with pytest.raises(RuntimeError, match="run_technical_preflight is disabled"):
        module.validate_scientific_contract(config, run_stage="technical_preflight")
    with pytest.raises(RuntimeError, match="full OOF is not separately approved"):
        module.validate_scientific_contract(config, run_stage="full_oof")
    changed = deepcopy(config)
    changed["model"]["soft_sticky"]["base_switching_length_md_ft"] = 999.0
    with pytest.raises(ValueError, match="contract changed"):
        module.validate_scientific_contract(changed)


def test_k16_schedule_ports_only_relative_geometry_rate(module):
    known_rows = 30
    suffix_rows = 32
    total_rows = known_rows + suffix_rows
    horizontal = pd.DataFrame(
        {
            "MD": np.arange(total_rows, dtype=float),
            "Z": np.zeros(total_rows),
            "GR": np.ones(total_rows),
            "TVT_input": np.r_[
                100.0 + np.arange(known_rows, dtype=float),
                [np.nan] * suffix_rows,
            ],
        }
    )
    segment = module.k16_segment_ids(suffix_rows, 16)
    step_rate = 2.0 + 0.1 * segment
    geometry_tvt = np.empty(suffix_rows, dtype=float)
    geometry_tvt[0] = 200.0
    for row in range(1, suffix_rows):
        geometry_tvt[row] = geometry_tvt[row - 1] + step_rate[row]
    geometry = pd.DataFrame(
        {
            "well_id": "well_a",
            "row_idx": np.arange(known_rows, total_rows, dtype=np.int32),
            "suffix_offset": np.arange(suffix_rows, dtype=np.int32),
            "fold": np.zeros(suffix_rows, dtype=np.int8),
            "tvt_geop": geometry_tvt,
        }
    )
    schedule, ledger, fallback = module.build_well_rate_schedule(
        "well_a",
        geometry,
        horizontal,
        k_segments=16,
    )
    assert fallback["prefix_rate"] == pytest.approx(1.0)
    assert fallback["fallback_segments"] == 0
    assert ledger.loc[0, "mu_rate"] == pytest.approx(1.0)
    assert ledger.loc[5, "geometry_delta_rate"] == pytest.approx(0.5)
    assert ledger.loc[5, "mu_rate"] == pytest.approx(1.5)
    assert not {
        "TVT",
        "true_tvt",
        "tvt_pred",
        "gr_delta",
        "error",
    }.intersection(schedule.columns)


def test_dense_joint_transition_is_row_stochastic_and_docking_is_soft(module):
    grid = np.array([0.0, 1.0, 2.0])
    rates = np.array([-0.1, 0.0, 0.1])
    transition, diagnostics = module.build_dense_joint_transition(
        grid,
        rates,
        delta_md=2.0,
        effective_delta_z=0.0,
        sig_r=0.04,
        sig_p=0.4,
        momentum=0.998,
        q_e_previous=0.5,
        q_e_current=0.6,
        switching_length_md_ft=1000.0,
        docking_sigma_ft=0.5,
    )
    np.testing.assert_allclose(transition.sum(axis=1), 1.0, atol=1.0e-14)
    assert diagnostics["transition_row_sum_max_abs_error"] <= 1.0e-14
    assert np.all((diagnostics["docking"] >= 0.0) & (diagnostics["docking"] <= 1.0))
    assert transition[0, 0] == pytest.approx(np.exp(-2.0 / 1000.0))
    assert np.count_nonzero(transition[0, 1:]) > 1
    assert np.all(transition[1:, 0] < diagnostics["hazard"] + 1.0e-15)


def brute_force_posterior(initial, transitions, log_emission):
    row_count, state_count = log_emission.shape
    path_probability = []
    paths = list(itertools.product(range(state_count), repeat=row_count))
    for path in paths:
        probability = initial[path[0]] * np.exp(log_emission[0, path[0]])
        for row in range(1, row_count):
            probability *= transitions[row - 1][path[row - 1], path[row]] * np.exp(
                log_emission[row, path[row]]
            )
        path_probability.append(probability)
    path_probability = np.asarray(path_probability, dtype=np.float64)
    partition = float(path_probability.sum())
    posterior = np.zeros((row_count, state_count), dtype=np.float64)
    for path, probability in zip(paths, path_probability, strict=True):
        for row, state in enumerate(path):
            posterior[row, state] += probability
    posterior /= partition
    return posterior, np.log(partition)


def test_dense_forward_backward_matches_exhaustive_path_enumeration(module):
    initial = np.array([0.3, 0.4, 0.3])
    transitions = [
        np.array(
            [
                [0.7, 0.2, 0.1],
                [0.1, 0.8, 0.1],
                [0.2, 0.3, 0.5],
            ]
        ),
        np.array(
            [
                [0.6, 0.3, 0.1],
                [0.2, 0.5, 0.3],
                [0.1, 0.2, 0.7],
            ]
        ),
    ]
    emission = np.log(
        np.array(
            [
                [0.6, 0.3, 0.1],
                [0.2, 0.5, 0.3],
                [0.1, 0.3, 0.6],
            ]
        )
    )
    expected, expected_log_partition = brute_force_posterior(
        initial,
        transitions,
        emission,
    )
    actual, actual_log_partition = module.exact_dense_forward_backward(
        initial,
        transitions,
        emission,
    )
    np.testing.assert_allclose(actual, expected, atol=1.0e-13)
    assert actual_log_partition == pytest.approx(expected_log_partition, abs=1.0e-13)


def dense_initial_after_first_h_step(
    module,
    *,
    grid,
    rates,
    delta_md,
    effective_delta_z,
    sig_r,
    sig_p,
    momentum,
    start_position_index,
    start_sig,
    initial_residual_rate,
    r0_sig,
):
    h_source = np.zeros((len(grid), len(rates)), dtype=np.float64)
    step = grid[1] - grid[0]
    for position in range(len(grid)):
        for rate in range(len(rates)):
            h_source[position, rate] = np.exp(
                -0.5 * (((position - start_position_index) * step) / start_sig) ** 2
                - 0.5 * ((rates[rate] - initial_residual_rate) / r0_sig) ** 2
            )
    h_source /= h_source.sum()
    h_destination = np.zeros_like(h_source)
    for position in range(len(grid)):
        for rate in range(len(rates)):
            for destination_rate, rate_probability in module.rate_transition_probabilities(
                rates,
                rate,
                delta_md,
                sig_r,
                momentum,
            ):
                for (
                    destination_position,
                    position_probability,
                ) in module.position_transition_probabilities(
                    grid,
                    grid[position],
                    rates[destination_rate],
                    delta_md,
                    effective_delta_z,
                    sig_p,
                ):
                    h_destination[destination_position, destination_rate] += (
                        h_source[position, rate] * rate_probability * position_probability
                    )
    h_destination /= h_destination.sum()
    return h_destination


def test_optimized_switching_kernel_matches_dense_joint_trellis(module):
    grid = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    rates = np.array([-0.1, 0.0, 0.1], dtype=np.float64)
    delta_md = np.array([1.0, 1.2, 0.8], dtype=np.float64)
    effective_delta_z = np.array([0.05, -0.10, 0.03], dtype=np.float64)
    q_e = np.array([0.8, 1.1, 1.0], dtype=np.float64)
    h_emission = np.array(
        [
            [-0.3, -0.1, -0.8],
            [-0.6, -0.2, -0.4],
            [-0.5, -0.3, -0.1],
        ],
        dtype=np.float64,
    )
    e_emission = np.array([-0.2, -0.4, -0.15], dtype=np.float64)
    controls = {
        "step": 1.0,
        "sig_r": 0.04,
        "sig_p": 0.4,
        "start_position_index": 1.0,
        "start_sig": 0.75,
        "initial_residual_rate": 0.0,
        "r0_sig": 0.08,
        "momentum": 0.998,
        "switching_length": 5.0,
        "docking_sigma": 0.7,
        "initial_e_probability": 0.5,
    }
    result = module._soft_sticky_forward_backward(
        h_emission,
        e_emission,
        grid,
        rates,
        delta_md,
        effective_delta_z,
        q_e,
        controls["step"],
        controls["sig_r"],
        controls["sig_p"],
        controls["start_position_index"],
        controls["start_sig"],
        controls["initial_residual_rate"],
        controls["r0_sig"],
        controls["momentum"],
        controls["switching_length"],
        controls["docking_sigma"],
        controls["initial_e_probability"],
    )
    gamma_e, h_position, _, _, joint_mean = result[:5]

    h_initial = dense_initial_after_first_h_step(
        module,
        grid=grid,
        rates=rates,
        delta_md=delta_md[0],
        effective_delta_z=effective_delta_z[0],
        sig_r=controls["sig_r"],
        sig_p=controls["sig_p"],
        momentum=controls["momentum"],
        start_position_index=controls["start_position_index"],
        start_sig=controls["start_sig"],
        initial_residual_rate=controls["initial_residual_rate"],
        r0_sig=controls["r0_sig"],
    )
    initial = np.r_[
        controls["initial_e_probability"],
        (1.0 - controls["initial_e_probability"]) * h_initial.ravel(),
    ]
    transitions = []
    for row in range(1, len(delta_md)):
        transition, _ = module.build_dense_joint_transition(
            grid,
            rates,
            delta_md=delta_md[row],
            effective_delta_z=effective_delta_z[row],
            sig_r=controls["sig_r"],
            sig_p=controls["sig_p"],
            momentum=controls["momentum"],
            q_e_previous=q_e[row - 1],
            q_e_current=q_e[row],
            switching_length_md_ft=controls["switching_length"],
            docking_sigma_ft=controls["docking_sigma"],
        )
        transitions.append(transition)
    emission = np.zeros((len(delta_md), 1 + len(grid) * len(rates)))
    emission[:, 0] = e_emission
    for row in range(len(delta_md)):
        emission[row, 1:] = np.repeat(h_emission[row], len(rates))
    dense, _ = module.exact_dense_forward_backward(initial, transitions, emission)
    dense_h_position = dense[:, 1:].reshape(len(delta_md), len(grid), len(rates)).sum(2)
    dense_joint_mean = dense[:, 0] * q_e + dense_h_position @ grid
    np.testing.assert_allclose(gamma_e, dense[:, 0], atol=2.0e-6)
    np.testing.assert_allclose(h_position, dense_h_position, atol=2.0e-6)
    np.testing.assert_allclose(joint_mean, dense_joint_mean, atol=2.0e-6)
    np.testing.assert_allclose(gamma_e + h_position.sum(axis=1), 1.0, atol=1.0e-12)
    assert result[-2] <= 1.0e-8
    assert result[-1] <= 1.0e-10


def test_gaussian_missing_gr_policy_and_joint_mean_are_finite(module):
    h_emission, e_emission = module.gaussian_emissions(
        np.array([10.0, np.nan]),
        np.array([0.0, 10.0, 20.0]),
        np.array([10.0, 0.0]),
        10.0,
    )
    assert np.isfinite(h_emission).all()
    assert np.isfinite(e_emission).all()
    assert e_emission[0] == pytest.approx(0.0)
    assert e_emission[1] == pytest.approx(0.0)


def test_preflight_selection_uses_only_fold_length_and_is_duplicate_free(module):
    rows = []
    for fold in range(5):
        for local in range(4):
            well = f"f{fold}_w{local}"
            length = 10 + fold * 4 + local
            rows.extend(
                {
                    "well_id": well,
                    "fold": fold,
                    "row_idx": row,
                    "suffix_offset": row,
                    "tvt_geop": float(row),
                }
                for row in range(length)
            )
    geometry = pd.DataFrame(rows)
    selection = module.select_preflight_wells(geometry)
    assert len(selection) == 16
    assert selection["well_id"].nunique() == 16
    assert (
        selection.loc[selection["selection_reason"].eq("fold_longest")]
        .groupby("fold")
        .size()
        .eq(3)
        .all()
    )
    assert "true_tvt" not in geometry


def test_target_free_reader_excludes_suffix_truth(module, tmp_path):
    pd.DataFrame(
        {
            "MD": [0.0, 1.0],
            "Z": [0.0, -1.0],
            "GR": [50.0, 51.0],
            "TVT_input": [100.0, np.nan],
            "TVT": [100.0, 100.2],
            "error": [0.0, 99.0],
        }
    ).to_csv(tmp_path / "well_a__horizontal_well.csv", index=False)
    pd.DataFrame({"TVT": [99.0, 101.0], "GR": [49.0, 52.0]}).to_csv(
        tmp_path / "well_a__typewell.csv",
        index=False,
    )
    ledger = module.RoleReadLedger()
    horizontal, typewell = module.load_target_free_well(
        "well_a",
        tmp_path,
        ledger,
    )
    assert list(horizontal.columns) == ["MD", "Z", "GR", "TVT_input"]
    assert list(typewell.columns) == ["TVT", "GR"]
    assert ledger.suffix_truth_rows_before_freeze == 0


def test_raw_identity_manifest_preserves_exp355_column_contract(module, tmp_path):
    pd.DataFrame(
        {
            "MD": [0.0, 1.0],
            "Z": [0.0, -1.0],
            "GR": [50.0, 51.0],
            "TVT_input": [100.0, np.nan],
        }
    ).to_csv(tmp_path / "well_a__horizontal_well.csv", index=False)
    pd.DataFrame({"TVT": [99.0, 101.0], "GR": [49.0, 52.0]}).to_csv(
        tmp_path / "well_a__typewell.csv",
        index=False,
    )
    expected_frame = pd.DataFrame(
        [
            {
                "well_id": "well_a",
                "horizontal_raw_sha256": module.sha256_file(
                    tmp_path / "well_a__horizontal_well.csv"
                ),
                "typewell_raw_sha256": module.sha256_file(tmp_path / "well_a__typewell.csv"),
            }
        ]
    )
    local_config = {
        "validation": {"expected_wells": 1},
        "data": {"expected_raw_well_identity_sha256": module.logical_frame_sha256(expected_frame)},
    }
    actual, report = module.validate_raw_identity(local_config, tmp_path)
    assert list(actual.columns) == [
        "well_id",
        "horizontal_raw_sha256",
        "typewell_raw_sha256",
    ]
    assert report["content_sha256"] == local_config["data"]["expected_raw_well_identity_sha256"]


def test_notebook_source_is_self_contained_and_canonical_is_adopted(config):
    source = TRAIN_SOURCE.read_text()
    assert "## 6. Soft-sticky transition and exact forward-backward helpers" in source
    assert "## 10. Kaggle CPU orchestration and generated artifacts" in source
    assert "__file__" not in source
    assert "from exact_hmm_smoother import" not in source
    assert "from exp394_" not in source
    assert "import exp394_" not in source
    assert config["implementation"]["canonical_notebook_adopted"] is True
    canonical = EXP_DIR / "exp394_soft_sticky_exp226_k16_branch_hmm_train.ipynb"
    canonical_text = canonical.read_text()
    assert "Design-only placeholder" not in canonical_text
    assert "Soft-sticky transition and exact forward-backward helpers" in canonical_text
