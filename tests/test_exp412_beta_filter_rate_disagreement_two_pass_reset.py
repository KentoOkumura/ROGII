from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.util
import inspect
import sys
import types
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

if importlib.util.find_spec("numba") is None:
    numba_stub = types.ModuleType("numba")
    numba_stub.__spec__ = importlib.machinery.ModuleSpec("numba", loader=None)

    def _njit(*args, **kwargs):
        del kwargs
        if args and callable(args[0]):
            return args[0]

        def decorator(function):
            return function

        return decorator

    numba_stub.njit = _njit
    numba_stub.prange = range
    numba_stub.set_num_threads = lambda threads: None
    numba_stub.get_num_threads = lambda: 1
    numba_stub.__version__ = "test-stub"
    sys.modules["numba"] = numba_stub

ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = (
    ROOT
    / "experiments"
    / "exp412_beta_filter_rate_disagreement_two_pass_reset"
)
TRAIN_SOURCE = (
    EXP_DIR
    / "exp412_beta_filter_rate_disagreement_two_pass_reset_"
    "compact_selfcontained_train.py"
)
INFERENCE_SOURCE = (
    EXP_DIR
    / "exp412_beta_filter_rate_disagreement_two_pass_reset_"
    "compact_selfcontained_inference.py"
)
BUILDER_SOURCE = EXP_DIR / "build_stage0_manifest.py"
CONFIG_PATH = EXP_DIR / "config.yaml"
MANIFEST_PATH = EXP_DIR / "assets" / "stage0_fixed32_manifest.csv"
PARENT_SOURCE = (
    ROOT
    / "experiments"
    / "exp209_exp072_exp205_joint_exact_parity_fast_cache_generation"
    / "exact_hmm_smoother.py"
)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def train():
    return load_module(TRAIN_SOURCE, "exp412_train_test")


@pytest.fixture(scope="module")
def inference():
    return load_module(INFERENCE_SOURCE, "exp412_inference_test")


@pytest.fixture(scope="module")
def parent():
    return load_module(PARENT_SOURCE, "exp412_exp209_reference")


@pytest.fixture(scope="module")
def config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())


def test_stage0_execution_is_explicitly_authorized_and_fixed(train, config):
    counts = train.validate_execution_contract(
        config,
        require_run_authorization=False,
    )
    assert counts == {
        "active_treatment_variants": 1,
        "stage_0_baseline_hmm_well_runs": 32,
        "stage_0_treatment_hmm_well_runs": 32,
        "stage_0_total_hmm_well_runs": 64,
        "parent_control_hmm_reruns": 32,
        "lightgbm_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
        "models": 0,
        "pf_well_runs": 0,
        "beam_well_runs": 0,
        "gpu_runs": 0,
    }
    assert config["design"]["implementation_enabled"] is True
    assert config["design"]["stage_0_execution_approved"] is True
    assert config["execution"]["kaggle_execution_authorized"] is True
    assert (
        train.validate_execution_contract(
            config,
            require_run_authorization=True,
        )
        == counts
    )


def test_saved_exp209_control_sha_is_full_and_consistent(config):
    expected = "8e2f42367b7b8b28e73094eae642c57c75dc8a7ebcfbc3826b0f2067b37f7ae5"
    assert len(expected) == 64
    assert config["data"]["parent_hmm_cache_decompressed_sha256"] == expected
    assert (
        config["data"]["exp209_saved_control"]["expected_decompressed_sha256"]
        == expected
    )


def test_saved_parent_parity_uses_exp209_float32_storage_contract(train):
    saved = np.asarray([12_000.0, 12_001.0], dtype=np.float32)
    within_same_storage_values = saved.astype(np.float64) + np.asarray(
        [0.0004, -0.0004]
    )
    assert (
        train.saved_float32_parity_max_abs_diff(
            within_same_storage_values,
            saved,
        )
        == 0.0
    )

    one_ulp_changed = saved.copy()
    one_ulp_changed[0] = np.nextafter(
        one_ulp_changed[0],
        np.float32(np.inf),
    )
    assert (
        train.saved_float32_parity_max_abs_diff(
            one_ulp_changed,
            saved,
        )
        > 1.0e-5
    )


def test_contract_rejects_stage1_inference_or_missing_parent_rerun(train, config):
    broken = deepcopy(config)
    broken["design"]["stage_1_execution_approved"] = True
    with pytest.raises(ValueError, match="Stage 1"):
        train.validate_execution_contract(broken, require_run_authorization=False)

    broken = deepcopy(config)
    broken["design"]["inference_enabled"] = True
    with pytest.raises(ValueError, match="inference"):
        train.validate_execution_contract(broken, require_run_authorization=False)

    broken = deepcopy(config)
    broken["execution"]["parent_control_regeneration_stage_0"] = False
    with pytest.raises(ValueError, match="regenerate"):
        train.validate_execution_contract(broken, require_run_authorization=False)


def test_scientific_contract_pins_two_pass_trigger_and_single_change(train, config):
    contract = train.validate_scientific_contract(config)
    assert contract["parent_hmm"]["sig_r"] == 0.002
    assert contract["parent_hmm"]["sig_p"] == 0.02
    assert contract["parent_hmm"]["mom"] == 0.998
    assert contract["trigger"] == {
        "statistic": "standardized_smoothed_minus_filtered_rate",
        "denominator_floor": 0.005,
        "absolute_z_threshold": 2.0,
        "rolling_window_rows": 16,
        "qualifying_rows_min": 8,
        "same_sign_fraction_min": 0.75,
        "tie_policy": "inactive",
        "freeze_before_truth": True,
        "recompute_from_treatment": False,
    }
    assert contract["treatment"]["stay_mass_transfer_fraction"] == 0.10
    assert (
        contract["treatment"]["first_affected_transition"]
        == "transition_entering_active_row"
    )

    broken = deepcopy(config)
    broken["model"]["trigger"]["absolute_z_threshold"] = 1.9
    with pytest.raises(ValueError, match="trigger contract changed"):
        train.validate_scientific_contract(broken)


def test_fixed32_manifest_is_sha_pinned_cause_balanced_and_unique(config):
    observed = hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest()
    assert observed == config["data"]["stage_0_manifest"]["expected_sha256"]
    manifest = pd.read_csv(MANIFEST_PATH, dtype={"well": str})
    assert len(manifest) == 32
    assert manifest["well"].nunique() == 32
    assert manifest["role"].value_counts().to_dict() == {
        "control": 16,
        "backward_cause": 8,
        "forward_cause": 8,
    }
    for role in ("backward_cause", "forward_cause"):
        assert set(manifest.loc[manifest["role"].eq(role), "fold"]) == set(range(5))
    assert set(
        manifest.loc[manifest["role"].eq("control"), "quartile_match_distance"]
        .astype(int)
    ) <= {0, 1}


def test_manifest_builder_keeps_truth_and_error_out_of_control_matching():
    source = BUILDER_SOURCE.read_text()
    assert 'usecols=["GR", "TVT_input"]' in source
    assert '"truth_columns_read_for_matching": []' in source
    assert '"error_columns_read_for_matching": []' in source
    assert '"cause_membership_passed_to_hmm": False' in source


def test_directional_rate_kernel_moves_only_stay_mass_and_keeps_edge_noop(train):
    rates = np.linspace(-0.10, 0.10, 41)
    base = train.rate_kernel_probabilities(rates, 12.0, 0.002, 0.998, 0, 0.10)
    positive = train.rate_kernel_probabilities(
        rates, 12.0, 0.002, 0.998, 1, 0.10
    )
    negative = train.rate_kernel_probabilities(
        rates, 12.0, 0.002, 0.998, -1, 0.10
    )
    interior = 20
    moved = 0.10 * base[interior, 1]
    assert positive[interior, 2] == pytest.approx(base[interior, 2] + moved)
    assert positive[interior, 1] == pytest.approx(base[interior, 1] - moved)
    assert positive[interior, 0] == pytest.approx(base[interior, 0])
    assert negative[interior, 0] == pytest.approx(base[interior, 0] + moved)
    assert negative[interior, 1] == pytest.approx(base[interior, 1] - moved)
    assert negative[interior, 2] == pytest.approx(base[interior, 2])
    np.testing.assert_allclose(positive[-1], base[-1], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(negative[0], base[0], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(positive.sum(axis=1), base.sum(axis=1), atol=1e-14)
    np.testing.assert_allclose(negative.sum(axis=1), base.sum(axis=1), atol=1e-14)


def test_beta_filter_schedule_uses_inclusive_16_row_window_and_sign_majority(
    train,
    config,
):
    trigger = config["model"]["trigger"]
    filtered = np.zeros(24)
    std = np.zeros(24)
    smoothed = np.full(24, 0.01)
    schedule = train.beta_filter_activation_schedule(
        filtered,
        std,
        smoothed,
        trigger,
    )
    assert not np.any(schedule["active_direction"][:7])
    assert np.all(schedule["active_direction"][7:] == 1)
    assert schedule["qualifying_count"][7] == 8
    assert schedule["majority_fraction"][7] == pytest.approx(1.0)

    tied = np.r_[np.full(8, 0.01), np.full(8, -0.01)]
    tied_schedule = train.beta_filter_activation_schedule(
        np.zeros(16),
        np.zeros(16),
        tied,
        trigger,
    )
    assert tied_schedule["active_direction"][-1] == 0
    assert tied_schedule["majority_fraction"][-1] == pytest.approx(0.5)


def test_zero_schedule_matches_independent_exp209_reference(
    train,
    parent,
    config,
):
    rng = np.random.default_rng(412)
    row_count = 24
    positions = 13
    emission = rng.normal(0.0, 0.3, size=(row_count, positions)).astype(np.float32)
    dm = 1.0 + (np.arange(row_count, dtype=np.float64) % 5) * 0.2
    dz = 0.25 * np.sin(np.arange(row_count, dtype=np.float64) / 7.0)
    rates = np.linspace(-0.06, 0.06, 9, dtype=np.float64)
    prepared = {
        "emission_ll": emission,
        "dm": dm,
        "dz": dz,
        "grid": 11_900.0 + np.arange(positions) * 0.35,
        "rates": rates,
        "start_p": 6.0,
        "r0": 0.0,
        "eval_index": np.arange(row_count),
    }
    observed = train.run_hmm_pass(
        prepared,
        config["model"]["parent_hmm"],
        config["model"]["treatment"],
        frozen_direction=np.zeros(row_count, dtype=np.int8),
    )
    reference_position, reference_loglik = parent._hmm2_fb(
        emission,
        dm,
        dz,
        0.35,
        rates,
        0.002,
        0.02,
        6.0,
        0.75,
        0.0,
        0.01,
        1.0,
        0.998,
    )
    reference_mean = reference_position @ prepared["grid"]
    np.testing.assert_allclose(
        observed["posterior_mean"],
        reference_mean,
        rtol=0.0,
        atol=2e-7,
    )
    assert abs(observed["log_likelihood"] - reference_loglik) <= 2e-6
    assert np.all(observed["filtered_rate_std"] >= 0.0)
    assert np.all(observed["smoothed_rate_mean"] >= rates.min())
    assert np.all(observed["smoothed_rate_mean"] <= rates.max())


def test_frozen_schedule_is_the_only_treatment_input(train, config):
    rows = 14
    positions = 11
    prepared = {
        "emission_ll": np.vstack(
            [
                -0.5
                * (
                    (
                        np.linspace(-1.0, 1.0, positions)
                        - 0.5 * np.sin(index / 2.0)
                    )
                    / 0.35
                )
                ** 2
                for index in range(rows)
            ]
        ).astype(np.float32),
        "dm": np.full(rows, 15.0),
        "dz": np.linspace(-0.2, 0.4, rows),
        "grid": 11_900.0 + np.arange(positions) * 0.35,
        "rates": np.linspace(-0.10, 0.10, 41),
        "start_p": 5.0,
        "r0": 0.0,
        "eval_index": np.arange(rows),
    }
    zero = train.run_hmm_pass(
        prepared,
        config["model"]["parent_hmm"],
        config["model"]["treatment"],
        frozen_direction=np.zeros(rows, dtype=np.int8),
    )
    active = np.zeros(rows, dtype=np.int8)
    active[5:10] = 1
    treatment = train.run_hmm_pass(
        prepared,
        config["model"]["parent_hmm"],
        config["model"]["treatment"],
        frozen_direction=active,
    )
    assert zero["prediction_sha256"] != treatment["prediction_sha256"]
    np.testing.assert_array_equal(treatment["frozen_direction"], active)
    with pytest.raises(ValueError, match="only -1, 0, or 1"):
        train.run_hmm_pass(
            prepared,
            config["model"]["parent_hmm"],
            config["model"]["treatment"],
            frozen_direction=np.full(rows, 2, dtype=np.int8),
        )


def test_truth_and_cause_are_blocked_until_all_two_pass_predictions_freeze(train):
    ledger = train.LeakageLedger(expected_wells=2)
    ledger.freeze("a")
    with pytest.raises(RuntimeError, match="before all fixed32"):
        ledger.record_truth_late(10)
    with pytest.raises(RuntimeError, match="before all fixed32"):
        ledger.record_episode_late(2)
    ledger.freeze("b")
    ledger.record_truth_late(10)
    ledger.record_episode_late(2)
    assert ledger.truth_rows_before_all_freeze == 10
    assert ledger.episode_rows_before_all_freeze == 2


def test_inference_is_fail_closed(inference, config):
    assert inference.validate_disabled_inference(config) == {
        "stage_1_execution_approved": False,
        "inference_enabled": False,
        "submission_enabled": False,
    }
    with pytest.raises(RuntimeError, match="inference is disabled"):
        inference.run_inference()


def test_notebook_source_is_self_contained_truth_late_and_two_pass():
    source = TRAIN_SOURCE.read_text()
    assert "__file__" not in source
    assert "from settings import" not in source
    assert "from build_stage0_manifest import" not in source
    assert "cusum" not in source.lower()
    for heading in (
        "## 1. Imports and immutable execution contract",
        "## 3. Fixed32 manifest, saved parent, and target-free raw inputs",
        "## 5. Frozen-schedule two-pass forward/backward kernel",
        "## 6. Parent parity, trigger freeze, and target-free prediction freeze",
        "## 7. Truth-late direction, cause, and safety readout",
        "## 8. Stage 0 gates, generated artifacts, and metrics",
    ):
        assert heading in source
    parameters = inspect.signature(
        load_module(TRAIN_SOURCE, "exp412_train_signature").freeze_target_free_well
    ).parameters
    assert not {
        "truth",
        "error",
        "episodes",
        "cause",
        "fold",
        "role",
        "hidden_like_role",
    }.intersection(parameters)
