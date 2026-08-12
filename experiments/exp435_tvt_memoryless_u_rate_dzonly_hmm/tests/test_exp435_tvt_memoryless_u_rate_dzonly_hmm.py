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
    numba_stub.__version__ = "test-stub"
    sys.modules["numba"] = numba_stub

ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = ROOT / "experiments" / "exp435_tvt_memoryless_u_rate_dzonly_hmm"
TRAIN_SOURCE = (
    EXP_DIR
    / "exp435_tvt_memoryless_u_rate_dzonly_hmm_compact_selfcontained_train.py"
)
INFERENCE_SOURCE = (
    EXP_DIR
    / "exp435_tvt_memoryless_u_rate_dzonly_hmm_compact_selfcontained_inference.py"
)
CONFIG_PATH = EXP_DIR / "config.yaml"
MANIFEST_PATH = (
    ROOT
    / "experiments"
    / "exp411_predictive_filtered_rate_innovation_destick"
    / "assets"
    / "stage0_fixed32_manifest.csv"
)
EPISODE_PATH = (
    ROOT
    / "experiments"
    / "exp408_hmm_message_rate_basin_audit"
    / "assets"
    / "persistent_offset_episodes.csv"
)
CAUSE_PATH = (
    ROOT
    / "experiments"
    / "exp408_hmm_message_rate_basin_audit"
    / "artifacts"
    / "kaggle_v3"
    / "exp408_hmm_message_rate_basin_audit_episode_summary.csv"
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
    return load_module(TRAIN_SOURCE, "exp435_train_test")


@pytest.fixture(scope="module")
def inference():
    return load_module(INFERENCE_SOURCE, "exp435_inference_test")


@pytest.fixture(scope="module")
def config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())


def synthetic_prepared(rows: int = 18, positions: int = 23) -> dict:
    grid = 11_900.0 + np.arange(positions, dtype=np.float64) * 0.35
    x = np.linspace(-1.0, 1.0, positions)
    emission = np.vstack(
        [
            -0.5 * ((x - 0.35 * np.sin(row / 4.0)) / 0.42) ** 2
            for row in range(rows)
        ]
    ).astype(np.float32)
    return {
        "emission_ll": emission,
        "dm": 1.0 + (np.arange(rows, dtype=np.float64) % 4) * 0.25,
        "dz": 0.18 * np.cos(np.arange(rows, dtype=np.float64) / 5.0),
        "grid": grid,
        "rates": np.linspace(-0.10, 0.10, 41, dtype=np.float64),
        "start_p": 11.0,
        "r0": 0.04,
        "eval_index": np.arange(rows, dtype=np.int64),
    }


def test_stage0_is_completed_and_execution_is_relocked(train, config):
    counts = train.validate_execution_contract(
        config,
        require_run_authorization=False,
    )
    assert counts == {
        "active_scientific_variants": 2,
        "stage_0_treatment_variants": 2,
        "stage_0_wells_per_treatment": 32,
        "stage_0_treatment_hmm_well_runs": 64,
        "parent_control_hmm_reruns_stage_0": 0,
        "stage_1_max_treatment_variants": 2,
        "stage_1_wells_per_treatment": 773,
        "stage_1_max_treatment_hmm_well_runs": 1546,
        "parent_control_hmm_reruns_stage_1": 0,
        "lightgbm_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
        "models": 0,
        "pf_runs": 0,
        "beam_runs": 0,
        "gpu_runs": 0,
    }
    assert config["implementation"]["enabled"] is True
    assert config["design"]["implementation_authorized"] is True
    assert config["design"]["kaggle_stage_0_authorized"] is True
    assert config["design"]["kaggle_stage_0_completed"] is True
    assert config["execution"]["run_hmm"] is False
    assert config["execution"]["create_prediction"] is False
    with pytest.raises(RuntimeError, match="execution.run_hmm is false"):
        train.validate_execution_contract(
            config,
            require_run_authorization=True,
        )

    broken = deepcopy(config)
    broken["design"]["kaggle_stage_1_authorized"] = True
    with pytest.raises(ValueError, match="Stage 1"):
        train.validate_execution_contract(
            broken,
            require_run_authorization=False,
        )


def test_scientific_contract_pins_tvt_only_state_and_two_variants(train, config):
    contract = train.validate_scientific_contract(config)
    assert contract["active_scientific_variants"] == [
        "memoryless_41rate",
        "dz_only_r0",
    ]
    assert contract["state_contract"]["persistent_state"] == (
        "tvt_probability_distribution"
    )
    assert contract["rate_responsibility_persisted"] is False
    assert contract["dz_only_rates"] == [0.0]
    assert contract["memoryless_stationary_sd"] == pytest.approx(
        0.002 / np.sqrt(1.0 - 0.998**2),
        abs=1.0e-15,
    )

    broken = deepcopy(config)
    broken["model"]["variants"]["memoryless_41rate"][
        "uses_init_rate_as_weight_mean"
    ] = True
    with pytest.raises(ValueError, match="cannot center"):
        train.validate_scientific_contract(broken)


def test_stationary_rate_weights_are_symmetric_zero_centered(train, config):
    hmm = config["model"]["shared_hmm"]
    rates = np.linspace(-0.14, 0.14, 41)
    weights = train.stationary_rate_weights(
        rates,
        sig_r=hmm["sig_r"],
        mom=hmm["mom"],
    )
    np.testing.assert_allclose(weights, weights[::-1], atol=1.0e-15)
    assert weights.sum() == pytest.approx(1.0, abs=1.0e-14)
    assert weights @ rates == pytest.approx(0.0, abs=1.0e-15)
    assert weights[20] > weights[0]


def test_position_kernels_and_mixture_are_normalized(train, config):
    hmm = config["model"]["shared_hmm"]
    rates = np.linspace(-0.10, 0.10, 41)
    weights = train.stationary_rate_weights(
        rates,
        sig_r=hmm["sig_r"],
        mom=hmm["mom"],
    )
    _, per_rate, per_rate_error = train.position_edge_kernels(
        rates,
        1.25,
        0.17,
        hmm["step"],
        hmm["sig_p"],
    )
    np.testing.assert_allclose(per_rate.sum(axis=1), 1.0, atol=1.0e-14)
    assert per_rate_error <= 1.0e-14
    _, mixture, mixture_error = train.mixed_position_kernel(
        rates,
        weights,
        1.25,
        0.17,
        hmm["step"],
        hmm["sig_p"],
    )
    assert mixture.sum() == pytest.approx(1.0, abs=1.0e-14)
    assert mixture_error <= 1.0e-14


def test_dz_only_is_exact_delta_rate_special_case(train, config):
    prepared = synthetic_prepared()
    hmm = config["model"]["shared_hmm"]
    observed = train.run_hmm_variant(prepared, hmm, "dz_only_r0")
    explicit = train.run_tvt_only_hmm(
        prepared,
        hmm,
        rates=np.asarray([0.0]),
        rate_weights=np.asarray([1.0]),
    )
    np.testing.assert_array_equal(
        observed["posterior_mean"],
        explicit["posterior_mean"],
    )
    np.testing.assert_array_equal(
        observed["posterior_std"],
        explicit["posterior_std"],
    )
    assert observed["prediction_sha256"] == explicit["prediction_sha256"]
    assert observed["transition_row_sum_max_error"] <= 1.0e-14
    assert observed["posterior_normalization_max_error"] <= 1.0e-6


def test_memoryless_hmm_persists_only_tvt_distribution(train, config):
    prepared = synthetic_prepared(rows=14, positions=19)
    observed = train.run_hmm_variant(
        prepared,
        config["model"]["shared_hmm"],
        "memoryless_41rate",
    )
    assert observed["persistent_state_shape"] == (14, 19)
    assert observed["edge_rate_count"] == 41
    assert np.isfinite(observed["posterior_mean"]).all()
    assert np.isfinite(observed["posterior_std"]).all()
    assert observed["transition_row_sum_max_error"] <= 1.0e-14
    assert observed["posterior_normalization_max_error"] <= 1.0e-6
    source = inspect.getsource(train._tvt_only_forward_backward)
    assert "alpha = np.empty((time_count, position_count)" in source
    assert "position_count, rate_count" not in source


def test_variant_mechanism_gates_are_independent(train, config):
    roles = ["persistent"] * 16 + ["control"] * 16
    folds = [0, 1, 2, 3, 4] * 6 + [0, 1]
    well_rows: list[dict] = []
    for variant in ("memoryless_41rate", "dz_only_r0"):
        for index, (role, fold) in enumerate(zip(roles, folds, strict=True)):
            parent = 10.0
            candidate = 9.0 if variant == "memoryless_41rate" else 11.0
            if role == "control":
                candidate = parent
            well_rows.append(
                {
                    "well": f"w{index:02d}",
                    "role": role,
                    "fold": fold,
                    "variant": variant,
                    "rows": 100,
                    "parent_rmse_ft": parent,
                    "variant_rmse_ft": candidate,
                    "rmse_delta_vs_parent_ft": candidate - parent,
                    "improved_vs_parent": candidate < parent,
                }
            )
    episode_rows: list[dict] = []
    for variant in ("memoryless_41rate", "dz_only_r0"):
        for index in range(16):
            parent_sse = 100.0
            variant_sse = (
                80.0 if variant == "memoryless_41rate" else 120.0
            )
            episode_rows.append(
                {
                    "episode_id": f"e{index:02d}",
                    "well": f"w{index:02d}",
                    "fold": folds[index],
                    "cause": "forward_transition_prior_hysteresis",
                    "variant": variant,
                    "rows": 10,
                    "parent_sse": parent_sse,
                    "variant_sse": variant_sse,
                }
            )
    well_metrics = pd.DataFrame(well_rows)
    episodes = pd.DataFrame(episode_rows)
    mechanism = config["validation"]["stage_0"]["mechanism"]
    memoryless = train.evaluate_variant_mechanism_gates(
        variant="memoryless_41rate",
        episode_readout=episodes,
        well_metrics=well_metrics,
        mechanism_config=mechanism,
        forward_cause="forward_transition_prior_hysteresis",
    )
    dz_only = train.evaluate_variant_mechanism_gates(
        variant="dz_only_r0",
        episode_readout=episodes,
        well_metrics=well_metrics,
        mechanism_config=mechanism,
        forward_cause="forward_transition_prior_hysteresis",
    )
    assert memoryless["all_mechanism_gates_pass"] is True
    assert dz_only["all_mechanism_gates_pass"] is False


def test_truth_role_fold_and_episode_reads_are_locked_until_all_freeze(train):
    ledger = train.LeakageLedger(expected_wells=2)
    ledger.freeze("a")
    with pytest.raises(RuntimeError, match="truth"):
        ledger.record_truth_late(3)
    with pytest.raises(RuntimeError, match="episodes"):
        ledger.record_episode_late(1)
    with pytest.raises(RuntimeError, match="role/fold"):
        ledger.record_role_fold_late(2)
    ledger.freeze("b")
    ledger.record_truth_late(3)
    ledger.record_episode_late(1)
    ledger.record_role_fold_late(2)
    assert ledger.truth_rows_before_all_freeze == 3
    assert ledger.episode_rows_before_all_freeze == 1
    assert ledger.role_fold_rows_before_all_freeze == 2


def test_fixed_inputs_are_sha_pinned(config):
    assert hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest() == (
        config["data"]["stage_0_manifest"]["expected_sha256"]
    )
    assert hashlib.sha256(EPISODE_PATH.read_bytes()).hexdigest() == (
        config["data"]["persistent_episodes"]["expected_sha256"]
    )
    assert hashlib.sha256(CAUSE_PATH.read_bytes()).hexdigest() == (
        config["data"]["exp408_episode_causes"]["expected_sha256"]
    )
    causes = pd.read_csv(CAUSE_PATH, usecols=["episode_id", "cause"])
    assert not causes["episode_id"].duplicated().any()
    assert "forward_transition_prior_hysteresis" in set(causes["cause"])


def test_inference_is_fail_closed(inference, config):
    contract = inference.validate_inference_disabled(config)
    assert contract["implementation_complete"] is True
    assert contract["stage0_completed"] is True
    assert contract["stage0_all_gates_pass"] is False
    assert contract["stage1_authorized"] is False
    assert contract["inference_authorized"] is False
    assert contract["submission_authorized"] is False
    with pytest.raises(RuntimeError, match="inference is disabled"):
        inference.run_inference(config)


def test_notebook_sources_are_self_contained_and_notebook_safe():
    train_source = TRAIN_SOURCE.read_text()
    inference_source = INFERENCE_SOURCE.read_text()
    assert "__file__" not in train_source
    assert "__file__" not in inference_source
    assert "from settings import" not in train_source
    assert "from settings import" not in inference_source
    assert "# ## 5. TVT-only forward-backward" in train_source
    assert "# ## 8. Stage 0 gates" in train_source
