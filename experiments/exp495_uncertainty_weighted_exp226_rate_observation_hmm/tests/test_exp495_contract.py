from __future__ import annotations

import copy
import importlib.util
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
SOURCE = (
    EXPERIMENT_DIR
    / "exp495_uncertainty_weighted_exp226_rate_observation_hmm_compact_selfcontained_train.py"
)
os.environ["EXP495_IMPORT_ONLY"] = "1"
SPEC = importlib.util.spec_from_file_location("exp495_stage0b", SOURCE)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def safe_horizontal(rows: int = 180, known_rows: int = 150) -> pd.DataFrame:
    md = np.arange(rows, dtype=np.float64) * 2.0
    z = -0.05 * md
    tvt = 0.20 * md - z
    tvt_input = tvt.copy()
    tvt_input[known_rows:] = np.nan
    return pd.DataFrame(
        {
            "X": np.arange(rows, dtype=np.float64),
            "Y": np.zeros(rows, dtype=np.float64),
            "Z": z,
            "MD": md,
            "TVT_input": tvt_input,
        }
    )


def test_repository_config_satisfies_stage0b_override_contract() -> None:
    config = MODULE.read_yaml(EXPERIMENT_DIR / "config.yaml")
    counts = MODULE.validate_scientific_contract(config)
    assert counts == {
        "scientific_variants": 1,
        "hmm_well_runs": 32,
        "models": 0,
        "trained_folds": 0,
        "boosters": 0,
        "parent_control_reruns": 0,
        "gpu_runs": 0,
    }
    assert config["authorization"]["stage_0a_fail_closed_override_approved"] is True
    assert config["execution"]["run_stage_0b"] is False
    assert config["execution"]["run_stage_1"] is False
    assert config["data"]["exp355_reference"]["metrics_prediction_sha_key"] == "prediction_sha256"


def test_strict_exp226_allowlist_excludes_truth_and_final_prediction() -> None:
    config = MODULE.read_yaml(EXPERIMENT_DIR / "config.yaml")
    allowlist = set(config["data"]["exp226_geometry"]["allowed_geometry_fields"])
    forbidden = set(config["data"]["exp226_geometry"]["forbidden_fields"])
    assert allowlist == set(MODULE.SAFE_OOF_COLUMNS)
    assert not allowlist.intersection(forbidden)
    assert {"tvt_pred", "tvt_true", "TVT", "gr_delta", "error"}.issubset(forbidden)


def test_target_safe_frame_rejects_truth_columns() -> None:
    frame = safe_horizontal()
    MODULE.validate_target_safe_frame(frame)
    with pytest.raises(ValueError, match="forbidden"):
        MODULE.validate_target_safe_frame(frame.assign(TVT=0.0))


def test_last_contiguous_known_index_rejects_a_hole() -> None:
    values = np.array([1.0, 2.0, np.nan, 4.0, np.nan])
    with pytest.raises(ValueError, match="finite rows after"):
        MODULE.last_contiguous_known_index(values)


def test_prefix_mask_selects_last_128_valid_transitions_and_masks_future() -> None:
    config = MODULE.read_yaml(EXPERIMENT_DIR / "config.yaml")
    frame = safe_horizontal(rows=190, known_rows=150)
    masked, destinations, manifest = MODULE.build_prefix_mask("well", frame, config)
    assert len(destinations) == 128
    assert destinations[0] == 22
    assert destinations[-1] == 149
    assert manifest["replay_cut_row"] == 21
    assert masked.loc[22:, "TVT_input"].isna().all()
    assert frame.loc[22:149, "TVT_input"].notna().all()


def test_prefix_transition_selection_skips_invalid_md_step() -> None:
    frame = safe_horizontal(rows=190, known_rows=150)
    frame.loc[100, "MD"] = frame.loc[99, "MD"]
    _, destinations = MODULE.valid_prefix_transition_destinations(frame, maximum_transitions=128)
    assert 100 not in set(destinations)
    assert len(destinations) == 128


def test_coordinate_rate_adds_z_once_and_has_formula_parity() -> None:
    md = np.array([0.0, 2.0, 4.0])
    z = np.array([0.0, -1.0, -2.0])
    tvt = np.array([0.0, 3.0, 6.0])
    rate, valid, parity = MODULE.coordinate_step_rates(tvt, z, md)
    np.testing.assert_allclose(rate, [1.0, 1.0])
    assert valid.all()
    assert parity <= 1.0e-15


def test_robust_uncertainty_uses_centered_mad_without_bias_correction() -> None:
    result = MODULE.robust_prefix_uncertainty(
        [1.0, 2.0, 3.0, 100.0],
        minimum_valid=4,
        multiplier=1.4826,
        floor=0.002,
    )
    assert result["residual_median"] == 2.5
    assert result["residual_mad"] == 1.0
    assert result["sigma_226"] == pytest.approx(1.4826)
    assert result["observation_enabled"] is True
    assert "bias" not in result


def test_robust_uncertainty_falls_back_below_32_transitions() -> None:
    result = MODULE.robust_prefix_uncertainty(
        np.zeros(31), minimum_valid=32, multiplier=1.4826, floor=0.002
    )
    assert result["sigma_226"] == 0.002
    assert result["observation_enabled"] is False
    assert result["fallback_reason"] == "insufficient_valid_prefix_transitions"


def test_prefix_transition_replay_recovers_zero_residual_on_exact_geometry() -> None:
    config = MODULE.read_yaml(EXPERIMENT_DIR / "config.yaml")
    frame = safe_horizontal(rows=190, known_rows=150)
    masked, destinations, manifest = MODULE.build_prefix_mask("well", frame, config)
    cut = manifest["replay_cut_row"]
    md = frame["MD"].to_numpy(np.float64)
    z = frame["Z"].to_numpy(np.float64)
    exact_tvt = 0.20 * md - z
    path = exact_tvt[cut + 1 :]
    target = SimpleNamespace(n=len(masked) - cut - 1)
    transitions, uncertainty = MODULE.build_prefix_transition_rows(
        "well",
        0,
        frame,
        target,
        path,
        destinations,
        manifest,
        config,
    )
    assert len(transitions) == 128
    assert transitions["valid"].all()
    np.testing.assert_allclose(transitions["rate_residual"], 0.0, atol=1.0e-12)
    assert uncertainty["valid_transition_count"] == 128
    assert uncertainty["sigma_226"] == 0.002
    assert uncertainty["observation_enabled"] is True


def test_k16_segment_identity_matches_exp226_linspace_rule() -> None:
    actual = MODULE.k16_segment_ids(101, 16)
    edges = np.linspace(0.0, 101.0, 17)
    expected = np.clip(
        np.searchsorted(edges[1:], np.arange(1.0, 102.0), side="left"), 0, 15
    ).astype(np.int16)
    np.testing.assert_array_equal(actual, expected)


def test_suffix_schedule_preserves_relative_geometry_center() -> None:
    horizontal = safe_horizontal(rows=250, known_rows=150)
    row_idx = np.arange(150, 250, dtype=np.int32)
    md = horizontal.loc[row_idx, "MD"].to_numpy(np.float64)
    z = horizontal.loc[row_idx, "Z"].to_numpy(np.float64)
    # geometry U-rate is a constant 0.30 while the known-prefix parent rate is 0.20
    geop = 0.30 * md - z
    geometry = pd.DataFrame(
        {
            "well_id": "well",
            "row_idx": row_idx,
            "suffix_offset": np.arange(100, dtype=np.int32),
            "fold": np.zeros(100, dtype=np.int8),
            "tvt_geop": geop,
        }
    )
    uncertainty = pd.Series(
        {"sigma_226": 0.01, "observation_enabled": True, "fallback_reason": "none"}
    )
    schedule, ledger, fallback = MODULE.build_well_suffix_schedule(
        "well", geometry, horizontal, uncertainty, k_segments=16
    )
    np.testing.assert_allclose(schedule["mu_226"], 0.20, atol=1.0e-12)
    np.testing.assert_allclose(
        ledger.loc[ledger["valid_geometry_steps"] > 0, "geometry_segment_rate"],
        0.30,
    )
    assert schedule["formula_parity_max_abs"].max() <= 1.0e-10
    assert fallback["prefix_observation_enabled"] is True


def minimal_frozen() -> object:
    prefix = pd.DataFrame(
        {
            "well_id": ["a"],
            "fold": [0],
            "destination_row_idx": [1],
            "transition_rank": [1],
            "delta_md": [1.0],
            "observed_u_rate": [0.1],
            "geometry_u_rate": [0.1],
            "rate_residual": [0.0],
            "formula_parity_abs": [0.0],
            "valid": [True],
        }
    )
    uncertainty = pd.DataFrame(
        {
            "well_id": ["a"],
            "fold": [0],
            "official_last_known_row": [1],
            "replay_cut_row": [0],
            "selected_transition_count": [1],
            "valid_transition_count": [1],
            "residual_median": [0.0],
            "residual_mad": [0.0],
            "sigma_226": [0.002],
            "observation_enabled": [False],
            "fallback_reason": ["insufficient_valid_prefix_transitions"],
            "formula_parity_max_abs": [0.0],
        }
    )
    schedule = pd.DataFrame(
        {
            "well_id": ["a"],
            "row_idx": [2],
            "suffix_offset": [0],
            "fold": [0],
            "segment_id": [0],
            "md": [2.0],
            "z": [0.0],
            "delta_md": [1.0],
            "md_since": [1.0],
            "tvt_geop": [0.2],
            "parent_initial_rate": [0.1],
            "geometry_segment_rate": [0.1],
            "geometry_delta_rate": [0.0],
            "mu_226": [0.1],
            "sigma_226": [0.002],
            "observation_enabled": [False],
            "geometry_fallback": [False],
            "anchor_u": [0.1],
            "formula_parity_max_abs": [0.0],
        }
    )
    ledger = pd.DataFrame({"well_id": ["a"], "segment_id": [0], "value": [1.0]})
    return MODULE.FrozenStage0A(
        prefix_transitions=prefix,
        uncertainty=uncertainty,
        suffix_schedule=schedule,
        segment_ledger=ledger,
        fallback_summary=pd.DataFrame({"well_id": ["a"]}),
        prefix_transitions_sha256=MODULE.dataframe_content_sha256(
            prefix, MODULE.PREFIX_TRANSITION_COLUMNS
        ),
        uncertainty_sha256=MODULE.dataframe_content_sha256(uncertainty, MODULE.UNCERTAINTY_COLUMNS),
        suffix_schedule_sha256=MODULE.dataframe_content_sha256(
            schedule, MODULE.SCHEDULE_CONTENT_COLUMNS
        ),
        segment_ledger_sha256=MODULE.dataframe_content_sha256(ledger),
    )


def test_truth_late_join_guard_rejects_mutated_frozen_schedule() -> None:
    frozen = minimal_frozen()
    MODULE.require_unchanged_freeze(frozen)
    frozen.suffix_schedule.loc[0, "mu_226"] = 0.2
    with pytest.raises(RuntimeError, match="changed frozen inputs"):
        MODULE.require_unchanged_freeze(frozen)


def test_spearman_handles_ties_deterministically() -> None:
    value = MODULE.spearman_correlation([1.0, 1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0])
    assert value == pytest.approx(0.9486832980505138)


def test_sigma_halves_use_stable_rank_not_truth_threshold() -> None:
    segment = pd.DataFrame(
        {
            "well_id": ["d", "c", "b", "a"],
            "fold": [0, 0, 1, 1],
            "actual_u_rate": [0.0, 0.0, 0.0, 0.0],
            "geometry_u_rate": [4.0, 3.0, 2.0, 1.0],
            "abs_geometry_rate_error": [4.0, 3.0, 2.0, 1.0],
        }
    )
    uncertainty = pd.DataFrame(
        {
            "well_id": ["a", "b", "c", "d"],
            "fold": [1, 1, 0, 0],
            "sigma_226": [0.1, 0.2, 0.3, 0.4],
            "observation_enabled": [True, True, True, True],
        }
    )
    readout = MODULE.build_well_reliability_readout(segment, uncertainty)
    low = readout.loc[readout["sigma_half"] == "low", "well_id"].tolist()
    high = readout.loc[readout["sigma_half"] == "high", "well_id"].tolist()
    assert low == ["a", "b"]
    assert high == ["c", "d"]


def test_mechanism_metrics_fail_closed_when_every_well_falls_back() -> None:
    segment = pd.DataFrame(
        {
            "well_id": ["a"],
            "fold": [0],
            "actual_u_rate": [0.2],
            "geometry_u_rate": [0.3],
            "actual_relative_rate": [0.1],
            "baseline_relative_rate": [0.0],
            "exp355_relative_rate": [0.05],
        }
    )
    well = pd.DataFrame(
        {
            "well_id": ["a"],
            "fold": [0],
            "sigma_226": [0.002],
            "observation_enabled": [False],
            "suffix_geometry_rate_rmse": [0.1],
            "suffix_geometry_rate_mae": [0.1],
            "sigma_half": ["fallback"],
        }
    )
    mechanism, folds = MODULE.build_mechanism_metrics(segment, well)
    assert math.isnan(mechanism["sigma_vs_suffix_error_spearman"])
    assert mechanism["fallback_well_fraction"] == 1.0
    assert folds.empty


def test_gate_never_automatically_enables_stage_0b() -> None:
    config = MODULE.read_yaml(EXPERIMENT_DIR / "config.yaml")
    config = copy.deepcopy(config)
    technical = config["gates"]["stage_0a_technical"]
    technical.update({"expected_wells": 1, "expected_suffix_rows": 1, "expected_folds": 1})
    frozen = minimal_frozen()
    geometry = frozen.suffix_schedule[["well_id", "row_idx"]].copy()
    mechanism = {
        "sigma_vs_suffix_error_spearman": 1.0,
        "positive_spearman_folds": 5,
        "low_vs_high_rate_rmse_gain_fraction": 0.5,
        "low_sigma_schedule_gain_fraction": 0.5,
        "improving_schedule_folds": 5,
        "fallback_well_fraction": 0.0,
    }
    gate = MODULE.evaluate_stage_0a_gate(geometry, frozen, mechanism, config)
    assert gate["passed"] is True
    assert gate["automatic_stage_0b"] is False
    assert "separate_approval" in gate["decision"]


def test_rate_observation_kernel_uniform_factor_is_exact_parent() -> None:
    rates = np.linspace(-0.10, 0.10, 41, dtype=np.float64)
    parent, parent_error = MODULE.rate_observation_log_kernel(
        rates, 1.0, 0.002, 0.998, 0.05, 0.002, False
    )
    uniform, uniform_error = MODULE.rate_observation_log_kernel(
        rates, 1.0, 0.002, 0.998, 0.05, float("inf"), True
    )
    np.testing.assert_array_equal(parent, uniform)
    assert parent_error <= 1.0e-15
    assert uniform_error <= 1.0e-15


def test_rate_observation_kernel_row_normalizes_and_favors_mu_destination() -> None:
    rates = np.linspace(-0.10, 0.10, 41, dtype=np.float64)
    parent, _ = MODULE.rate_observation_log_kernel(
        rates, 1.0, 0.002, 0.998, 0.10, 0.002, False
    )
    observed, error = MODULE.rate_observation_log_kernel(
        rates, 1.0, 0.002, 0.998, 0.10, 0.002, True
    )
    np.testing.assert_allclose(np.exp(observed).sum(axis=1), 1.0, atol=1.0e-12)
    assert error <= 1.0e-12
    middle = 20
    assert np.exp(observed[middle, 2]) > np.exp(parent[middle, 2])


def test_stage0b_ledger_rejects_role_or_truth_before_all_predictions_freeze() -> None:
    ledger = MODULE.Stage0BLeakageLedger(expected_wells=2)
    ledger.freeze("a")
    with pytest.raises(RuntimeError, match="before all predictions froze"):
        ledger.record_late("role", 1)
    ledger.freeze("b")
    ledger.record_late("truth", 5)
    assert ledger.truth_rows_after_freeze == 5


def test_fixed32_scope_identity_comparison_is_dtype_independent() -> None:
    config = MODULE.read_yaml(EXPERIMENT_DIR / "config.yaml")
    identity, _ = MODULE.load_fixed32_identity(config)
    assert identity["prefix_rows"].dtype == np.dtype("int32")
    ledger = MODULE.Stage0BLeakageLedger(expected_wells=32)
    for well in identity["well"]:
        ledger.freeze(str(well))
    scope = MODULE.load_fixed32_scope_after_freeze(config, identity, ledger)
    assert len(scope) == 32
    assert scope["prefix_rows"].dtype != identity["prefix_rows"].dtype


def test_uniform_factor_full_hmm_parent_parity() -> None:
    config = MODULE.read_yaml(EXPERIMENT_DIR / "config.yaml")
    parity = MODULE.uniform_factor_parent_parity(config)
    assert parity["passed"] is True
    assert parity["maximum_posterior_mean_abs_diff_ft"] <= 1.0e-10


def test_run_guard_blocks_unapproved_stage_0a_before_io() -> None:
    config = MODULE.read_yaml(EXPERIMENT_DIR / "config.yaml")
    with pytest.raises(RuntimeError, match="not approved|must run on Kaggle"):
        MODULE.run_stage_0a(config)
