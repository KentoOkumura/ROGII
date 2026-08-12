from __future__ import annotations

import importlib.util
import os
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = (
    ROOT
    / "experiments"
    / "exp405_geometry_reinjected_interval_semimarkov_fusion"
)
SOURCE = EXP_DIR / (
    "exp405_geometry_reinjected_interval_semimarkov_fusion_"
    "compact_selfcontained_train.py"
)


def load_source():
    previous = os.environ.get("EXP405_IMPORT_ONLY")
    os.environ["EXP405_IMPORT_ONLY"] = "1"
    try:
        spec = importlib.util.spec_from_file_location("exp405_train_test", SOURCE)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        if previous is None:
            os.environ.pop("EXP405_IMPORT_ONLY", None)
        else:
            os.environ["EXP405_IMPORT_ONLY"] = previous


@pytest.fixture(scope="module")
def candidate():
    return load_source()


@pytest.fixture(scope="module")
def config():
    value = yaml.safe_load((EXP_DIR / "config.yaml").read_text())
    assert isinstance(value, dict)
    return value


def test_contract_counts_and_staged_execution_guards(
    candidate, config
):
    counts = candidate.validate_scientific_contract(config)
    assert counts == {
        "scientific_endpoints": 1,
        "negative_controls": 2,
        "reporting_folds": 5,
        "fixed_candidates": 12,
        "models": 0,
        "boosters": 0,
        "pf_runs": 0,
        "hmm_runs": 0,
        "beam_runs": 0,
        "parent_reruns": 0,
    }
    assert config["implementation"]["canonical_notebook_adopted"] is True
    assert config["execution"]["fixed16_preflight_passed"] is True
    assert config["execution"]["fixed16_preflight_summary_sha256"]
    assert config["current_test"]["implementation_enabled"] is False

    preflight = deepcopy(config)
    preflight["execution"]["run_stage"] = "fixed16_preflight"
    preflight["execution"]["run_fixed16_preflight"] = False
    with pytest.raises(RuntimeError, match="run_fixed16_preflight is disabled"):
        candidate.validate_scientific_contract(preflight)
    preflight["execution"]["run_fixed16_preflight"] = True
    preflight["execution"]["kaggle_execution_authorized"] = True
    candidate.validate_scientific_contract(preflight)

    full = deepcopy(preflight)
    full["execution"]["run_stage"] = "full_saved_oof"
    full["execution"]["run_full_saved_oof"] = True
    full["execution"]["full_saved_oof_approved"] = True
    full["execution"]["kaggle_execution_authorized"] = True
    full["execution"]["fixed16_preflight_passed"] = False
    with pytest.raises(RuntimeError, match="fixed16 preflight has not passed"):
        candidate.validate_scientific_contract(full)
    full["execution"]["fixed16_preflight_passed"] = True
    full["execution"]["fixed16_preflight_summary_sha256"] = "fixture"
    candidate.validate_scientific_contract(full)


def test_switch_transition_has_no_self_loop_and_unconditional_geometry_floor(
    candidate, config
):
    transition = candidate.build_switch_transition(config)
    np.testing.assert_allclose(transition.sum(axis=1), 1.0, atol=1.0e-15)
    np.testing.assert_array_equal(np.diag(transition), np.zeros(12))
    non_geometry = np.arange(12) != candidate.GEOMETRY_INDEX
    assert np.all(
        transition[non_geometry, candidate.GEOMETRY_INDEX] >= 0.10
    )
    source = SOURCE.read_text()
    core = source[source.index("def build_switch_transition") :]
    assert "docking" not in core[: core.index("def allowed_segment_ends")]


def enumerate_semimarkov(candidate, emission, config):
    blocks, states = emission.shape
    minimum = config["semimarkov"]["minimum_duration_blocks"]
    initial = candidate.segment_start_prior(config)
    transition = candidate.build_switch_transition(config)
    switch_mass = np.exp(
        -config["semimarkov"]["segment_switch_penalty"]["log_cost"]
    )
    total = 0.0
    marginal = np.zeros_like(emission, dtype=np.float64)

    def visit(start, previous, weight, labels):
        nonlocal total
        ends = candidate.allowed_segment_ends(start, blocks, minimum)
        for end in ends:
            duration_weight = 1.0 / len(ends)
            for state in range(states):
                if previous is None:
                    state_weight = initial[state]
                elif state == previous:
                    continue
                else:
                    state_weight = transition[previous, state] * switch_mass
                emission_weight = np.exp(emission[start:end, state].sum())
                new_weight = (
                    weight * duration_weight * state_weight * emission_weight
                )
                new_labels = labels + [(start, int(end), state)]
                if end == blocks:
                    total += new_weight
                    for left, right, label in new_labels:
                        marginal[left:right, label] += new_weight
                else:
                    visit(int(end), state, new_weight, new_labels)

    visit(0, None, 1.0, [])
    return np.log(total), marginal / total


def test_exact_semimarkov_matches_bruteforce_and_supports_final_short_segment(
    candidate, config
):
    emission = np.linspace(-1.2, -0.1, 4 * 12).reshape(4, 12)
    expected_logz, expected_posterior = enumerate_semimarkov(
        candidate, emission, config
    )
    actual = candidate.exact_interval_semimarkov(emission, config)
    assert actual.log_evidence == pytest.approx(expected_logz, abs=2.0e-13)
    np.testing.assert_allclose(
        actual.block_posterior,
        expected_posterior,
        atol=2.0e-13,
        rtol=0.0,
    )
    assert actual.normalization_abs_error_max <= 2.0e-13
    np.testing.assert_array_equal(
        candidate.allowed_segment_ends(3, 4, 2), np.array([4])
    )


def test_centered_rolling_is_full_window_and_block_local(candidate):
    values = np.arange(7, dtype=np.float64)
    actual = candidate.centered_full_window_mean(values, 3)
    np.testing.assert_allclose(actual[1:-1], np.arange(1, 6, dtype=np.float64))
    assert np.isnan(actual[[0, -1]]).all()
    left = candidate.centered_full_window_mean(values[:4], 3)
    right = candidate.centered_full_window_mean(values[4:], 3)
    assert np.isnan(left[-1])
    assert np.isnan(right[0])


def test_morphology_score_is_uniform_when_support_is_insufficient(
    candidate, config
):
    observed = np.linspace(0.0, 1.0, 32)
    paths = np.tile(np.linspace(100.0, 120.0, 32)[:, None], (1, 12))
    typewell_tvt = np.linspace(0.0, 300.0, 601)
    typewell_gr = np.sin(typewell_tvt / 11.0)
    score = candidate.score_block_morphology(
        observed, paths, typewell_tvt, typewell_gr, config
    )
    assert not score.eligible.any()
    np.testing.assert_allclose(
        np.exp(score.log_emission),
        np.full(12, 1.0 / 12.0),
        atol=1.0e-15,
    )


def test_morphology_score_prefers_matching_shape(candidate, config):
    rows = 256
    typewell_tvt = np.linspace(0.0, 600.0, 1201)
    typewell_gr = (
        0.02 * typewell_tvt
        + 18.0 * np.sin(typewell_tvt / 13.0)
        + 7.0 * np.cos(typewell_tvt / 29.0)
    )
    correct = np.linspace(120.0, 360.0, rows)
    observed = np.interp(correct, typewell_tvt, typewell_gr)
    paths = np.column_stack(
        [correct, *[correct + 90.0 + index for index in range(11)]]
    )
    score = candidate.score_block_morphology(
        observed, paths, typewell_tvt, typewell_gr, config
    )
    assert score.eligible.all()
    assert int(np.argmax(score.log_emission)) == 0
    assert np.exp(score.log_emission).sum() == pytest.approx(1.0)


def test_negative_controls_are_stable_and_preserve_declared_boundaries(
    candidate, config
):
    observed = np.arange(522, dtype=np.float64)
    observed[[3, 260, 519]] = np.nan
    blocks = [
        candidate.BlockSlice(0, 0, 256),
        candidate.BlockSlice(1, 256, 512),
        candidate.BlockSlice(2, 512, 522),
    ]
    circular_a, offset_a = candidate.circular_control(observed, "well-a", config)
    circular_b, offset_b = candidate.circular_control(observed, "well-a", config)
    np.testing.assert_array_equal(circular_a, circular_b)
    np.testing.assert_array_equal(np.isnan(circular_a), np.isnan(observed))
    assert offset_a == offset_b
    assert offset_a >= int(np.ceil(0.25 * np.isfinite(observed).sum()))

    permuted, order = candidate.block_permutation_control(
        observed, blocks, "well-a"
    )
    assert order == [1, 0]
    np.testing.assert_array_equal(permuted[:256], observed[256:512])
    np.testing.assert_array_equal(permuted[256:512], observed[:256])
    np.testing.assert_array_equal(permuted[512:], observed[512:])


def test_interpolation_is_convex_and_continuity_guarded(candidate):
    blocks = [
        candidate.BlockSlice(0, 0, 4),
        candidate.BlockSlice(1, 4, 8),
    ]
    posterior = np.zeros((2, 12), dtype=np.float64)
    posterior[0, 0] = 1.0
    posterior[1, 1] = 1.0
    weights = candidate.interpolate_block_weights(posterior, blocks, 8)
    np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1.0e-15)
    paths = np.column_stack(
        [np.arange(8, dtype=np.float64) + 10.0 * state for state in range(12)]
    )
    prediction = np.sum(weights * paths, axis=1)
    guards = candidate.prediction_guards(paths, weights, prediction)
    assert guards["convex_hull_coverage"] == 1.0
    assert guards["interpolation_guard_passed"] is True
    assert guards["physical_continuity_guard_passed"] is True


def test_constrained_oracle_allows_only_final_short_segment(candidate):
    blocks = [
        candidate.BlockSlice(0, 0, 2),
        candidate.BlockSlice(1, 2, 4),
        candidate.BlockSlice(2, 4, 6),
    ]
    paths = np.full((6, 12), 100.0)
    paths[:, 0] = 0.0
    paths[:, 1] = 10.0
    truth = np.array([0.0, 0.0, 0.0, 0.0, 10.0, 10.0])
    labels = candidate.constrained_oracle_labels(paths, truth, blocks, 2)
    np.testing.assert_array_equal(labels, np.array([0, 0, 1]))


def test_fixed16_selection_is_stable_and_fold_balanced(candidate, config):
    segments = [
        candidate.WellSegment(f"w{fold}-{index}", 0, 1, fold)
        for fold in range(5)
        for index in range(10)
    ]
    first = candidate.select_fixed16_segments(segments, config)
    second = candidate.select_fixed16_segments(list(reversed(segments)), config)
    assert [(item.well, item.fold) for item in first] == [
        (item.well, item.fold) for item in second
    ]
    counts = np.bincount([item.fold for item in first], minlength=5)
    np.testing.assert_array_equal(counts, np.array([4, 3, 3, 3, 3]))


def test_source_freezes_before_truth_or_hidden_roles_and_is_self_contained():
    source = SOURCE.read_text()
    freeze = source.index("freeze = freeze_target_free_generation(")
    truth = source.index("truth, truth_evidence, truth_ledger = load_truth_after_freeze(")
    hidden = source.index("load_hidden_like_sets_after_freeze(", truth)
    assert freeze < truth < hidden
    assert "__file__" not in source
    assert "from exp293_" not in source
    assert "import exp293_" not in source
    assert "current_test_implementation_eligible" in source
    assert "submission_created" in source
