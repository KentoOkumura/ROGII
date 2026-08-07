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
EXP_DIR = ROOT / "experiments" / "exp438_u_state_fixed_lattice_exact_hmm"
TRAIN_SOURCE = (
    EXP_DIR
    / "exp438_u_state_fixed_lattice_exact_hmm_compact_selfcontained_train.py"
)
INFERENCE_SOURCE = (
    EXP_DIR
    / "exp438_u_state_fixed_lattice_exact_hmm_compact_selfcontained_inference.py"
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
    return load_module(TRAIN_SOURCE, "exp438_train_test")


@pytest.fixture(scope="module")
def inference():
    return load_module(INFERENCE_SOURCE, "exp438_inference_test")


@pytest.fixture(scope="module")
def parent():
    return load_module(PARENT_SOURCE, "exp438_exp209_reference")


@pytest.fixture(scope="module")
def config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())


def synthetic_inputs(rows: int = 9) -> tuple[pd.DataFrame, pd.DataFrame]:
    prefix_rows = 8
    total = prefix_rows + rows
    md = np.arange(total, dtype=np.float64) * 10.0
    z = 8_000.0 + 0.3 * np.sin(np.arange(total, dtype=np.float64) / 2.5)
    visible_tvt = 12_000.0 + 0.02 * md - (z - z[0])
    tvt_input = visible_tvt.copy()
    tvt_input[prefix_rows:] = np.nan
    horizontal = pd.DataFrame(
        {
            "MD": md,
            "Z": z,
            "GR": 65.0 + 4.0 * np.sin(np.arange(total) / 3.0),
            "TVT_input": tvt_input,
        }
    )
    typewell_tvt = np.linspace(11_850.0, 12_150.0, 401)
    typewell = pd.DataFrame(
        {
            "TVT": typewell_tvt,
            "GR": 65.0 + 8.0 * np.sin((typewell_tvt - 12_000.0) / 18.0),
        }
    )
    return horizontal, typewell


def test_stage0_completed_fail_closed_and_rerun_is_locked(
    train,
    config,
):
    counts = train.validate_execution_contract(
        config,
        require_run_authorization=False,
    )
    assert counts == {
        "scientific_variants": 1,
        "reporting_folds": 5,
        "stage0_hmm_well_runs": 32,
        "stage1_max_hmm_well_runs": 773,
        "parent_control_hmm_reruns": 0,
        "fitted_ml_models": 0,
        "lightgbm_configs": 0,
        "trained_ml_folds": 0,
        "boosters": 0,
        "pf_runs": 0,
        "beam_runs": 0,
        "gpu_runs": 0,
    }
    assert (
        config["experiment"]["status"]
        == "stage0_fail_closed_v1"
    )
    assert config["runtime"]["implementation_approved"] is True
    assert config["runtime"]["run_approved"] is False
    assert config["execution"]["run_hmm"] is False
    assert config["implementation"]["kaggle_push_completed"] is True
    assert config["implementation"]["stage0_completed"] is True
    with pytest.raises(
        RuntimeError,
        match="does not authorize Kaggle execution",
    ):
        train.validate_execution_contract(
            config,
            require_run_authorization=True,
        )

    broken = deepcopy(config)
    broken["runtime"]["stage1_approved"] = True
    with pytest.raises(ValueError, match="Stage 1"):
        train.validate_execution_contract(
            broken,
            require_run_authorization=False,
        )


def test_scientific_contract_pins_joint_u_state_and_arrival_rate(train, config):
    contract = train.validate_scientific_contract(config)
    assert contract["candidate_state"] == ["u_position", "u_rate"]
    assert (
        contract["candidate_position"]["transition_mean_formula"]
        == "r_current*delta_MD"
    )
    assert (
        contract["fixed_from_exp209"]["rate_position_integration"]
        == "arrival_rate"
    )
    assert contract["fixed_from_exp209"]["position_kernel_cells"] == 5

    broken = deepcopy(config)
    broken["model"]["candidate_position"][
        "transition_mean_formula"
    ] = "0.5*(r_source+r_destination)*delta_MD"
    with pytest.raises(ValueError, match="fixed-U coordinate contract"):
        train.validate_scientific_contract(broken)


def test_preparation_anchors_one_fixed_u_grid_and_exact_row_tvt_view(
    train,
    config,
):
    horizontal, typewell = synthetic_inputs()
    prepared = train.prepare_hmm_inputs(
        horizontal,
        typewell,
        config["model"]["fixed_from_exp209"],
    )
    expected_u = (
        prepared["parent_tvt_grid"] + prepared["last_known_z"]
    )
    np.testing.assert_array_equal(prepared["u_grid"], expected_u)
    np.testing.assert_allclose(
        prepared["row_tvt_grid"],
        prepared["u_grid"][None, :] - prepared["z"][:, None],
        rtol=0.0,
        atol=0.0,
    )
    contract = train.coordinate_contract_from_prepared(prepared)
    assert contract["tvt_equals_u_minus_z_max_abs_ft"] <= 1.0e-12
    assert contract["transition_coordinate_identity_max_abs_ft"] <= 1.0e-12
    assert contract["emission_coordinate_identity_max_abs"] <= 1.0e-12


def test_fixed_u_joint_hmm_has_normalized_position_and_rate_posteriors(
    train,
    config,
):
    horizontal, typewell = synthetic_inputs(rows=6)
    hmm = config["model"]["fixed_from_exp209"]
    prepared = train.prepare_hmm_inputs(horizontal, typewell, hmm)
    result = train.run_fixed_u_hmm(prepared, hmm)
    assert result["posterior_position"].shape[0] == 6
    assert result["posterior_rate"].shape == (6, 41)
    np.testing.assert_allclose(
        result["posterior_position"].sum(axis=1),
        1.0,
        atol=1.0e-8,
    )
    np.testing.assert_allclose(
        result["posterior_rate"].sum(axis=1),
        1.0,
        atol=1.0e-8,
    )
    assert result["readout_identity_max_abs_ft"] <= 1.0e-12
    assert result["transition_row_sum_max_error"] <= 1.0e-10
    assert result["posterior_normalization_max_error"] <= 1.0e-8
    assert np.isfinite(result["mean_tvt"]).all()
    assert np.isfinite(result["rate_mean"]).all()


def test_constant_z_parent_parity_contract(train, config):
    contract = train.constant_z_parent_parity_contract(
        config["model"]["fixed_from_exp209"]
    )
    assert contract["pass"] is True
    assert contract["prediction_max_abs_ft"] <= 1.0e-6
    assert contract["position_posterior_max_abs"] <= 1.0e-6
    assert contract["rate_posterior_max_abs"] <= 1.0e-6
    assert contract["log_likelihood_abs"] <= 1.0e-6


def test_generic_fixed_lattice_kernel_matches_independent_exp209_parent(
    train,
    parent,
    config,
):
    rng = np.random.default_rng(438)
    row_count = 12
    position_count = 15
    emission = rng.normal(
        0.0,
        0.25,
        size=(row_count, position_count),
    ).astype(np.float32)
    dm = 1.0 + (np.arange(row_count, dtype=np.float64) % 4) * 0.2
    dz = 0.2 * np.sin(np.arange(row_count, dtype=np.float64) / 3.0)
    rates = np.linspace(-0.04, 0.04, 9, dtype=np.float64)
    hmm = config["model"]["fixed_from_exp209"]
    common = (
        emission,
        dm,
        dz,
        float(hmm["position_grid_step_ft"]),
        rates,
        float(hmm["sig_r"]),
        float(hmm["sig_p"]),
        7.1,
        float(hmm["start_sigma_ft"]),
        0.01,
        float(hmm["initial_rate_sigma"]),
        float(hmm["emission_lambda"]),
        float(hmm["momentum"]),
    )
    parent_position, parent_loglik = parent._hmm2_fb(*common)
    observed_position, _, observed_loglik, _ = (
        train._hmm2_fb_fixed_lattice(*common)
    )
    np.testing.assert_allclose(
        observed_position,
        parent_position,
        rtol=0.0,
        atol=2.0e-7,
    )
    assert abs(float(observed_loglik) - float(parent_loglik)) <= 2.0e-6


def test_exhaustive_small_path_reference_matches_joint_kernel(train, config):
    contract = train.brute_force_small_reference_contract(
        config["model"]["fixed_from_exp209"]
    )
    assert contract["pass"] is True
    assert contract["position_posterior_max_abs"] <= 1.0e-6
    assert contract["rate_posterior_max_abs"] <= 1.0e-6
    assert contract["log_likelihood_abs"] <= 1.0e-6


def test_quantization_ledger_uses_same_rate_posterior_for_both_coordinates(
    train,
):
    rows = 4
    rates = np.asarray([-0.05, 0.0, 0.05], dtype=np.float64)
    posterior_rate = np.asarray(
        [
            [0.2, 0.6, 0.2],
            [0.1, 0.7, 0.2],
            [0.3, 0.4, 0.3],
            [0.2, 0.5, 0.3],
        ],
        dtype=np.float64,
    )
    ledger = train.transition_quantization_ledger(
        dm=np.asarray([10.0, 11.0, 12.0, 13.0]),
        dz=np.asarray([0.11, -0.17, 0.23, -0.31]),
        rates=rates,
        posterior_rate=posterior_rate,
        step=0.35,
        sig_p=0.02,
        row_idx=np.arange(100, 100 + rows),
    )
    assert len(ledger) == rows
    assert ledger["nontrivial_z_phase"].all()
    assert np.isfinite(
        ledger[
            "parent_posterior_weighted_abs_quantization_bias_ft"
        ]
    ).all()
    assert np.isfinite(
        ledger[
            "candidate_posterior_weighted_abs_quantization_bias_ft"
        ]
    ).all()


def test_truth_role_fold_and_episode_reads_are_locked_until_all_freeze(train):
    ledger = train.LeakageLedger(expected_wells=2)
    ledger.freeze("a")
    with pytest.raises(RuntimeError, match="truth"):
        ledger.record_truth_late(3)
    with pytest.raises(RuntimeError, match="role/fold"):
        ledger.record_role_fold_late(2)
    with pytest.raises(RuntimeError, match="episodes"):
        ledger.record_episode_late(1)
    ledger.freeze("b")
    ledger.record_truth_late(3)
    ledger.record_role_fold_late(2)
    ledger.record_episode_late(1)
    assert ledger.truth_rows_before_all_freeze == 3
    assert ledger.role_fold_rows_before_all_freeze == 2
    assert ledger.episode_rows_before_all_freeze == 1


def test_fixed_inputs_are_sha_pinned(config):
    assert hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest() == (
        config["data"]["fixed32_manifest"]["expected_sha256"]
    )
    assert hashlib.sha256(EPISODE_PATH.read_bytes()).hexdigest() == (
        config["data"]["persistent_episodes"]["expected_sha256"]
    )
    assert hashlib.sha256(CAUSE_PATH.read_bytes()).hexdigest() == (
        config["data"]["exp408_episode_causes"]["expected_sha256"]
    )
    assert (
        config["data"]["exp209_saved_control"]["expected_decompressed_sha256"]
        == "8e2f42367b7b8b28e73094eae642c57c75dc8a7ebcfbc3826b0f2067b37f7ae5"
    )


def test_inference_is_fail_closed(inference, config):
    contract = inference.validate_inference_disabled(config)
    assert contract == {
        "implementation_approved": True,
        "stage0_run_approved": False,
        "stage1_approved": False,
        "inference_enabled": False,
        "submission_enabled": False,
        "create_submission": False,
    }
    with pytest.raises(RuntimeError, match="inference is disabled"):
        inference.run_inference(config)


def test_notebook_sources_are_self_contained_and_notebook_safe(train):
    train_source = TRAIN_SOURCE.read_text()
    inference_source = INFERENCE_SOURCE.read_text()
    assert "__file__" not in train_source
    assert "__file__" not in inference_source
    assert "from settings import" not in train_source
    assert "from settings import" not in inference_source
    assert "prange" not in train_source
    for heading in (
        "## 1. Imports and immutable execution contract",
        "## 3. Fixed32 manifest, saved parent, and target-free raw inputs",
        "## 4. Fixed-U coordinate and exp209 input preparation",
        "## 5. Joint fixed-lattice exact forward-backward HMM",
        "## 6. Numerical contracts and target-free prediction freeze",
        "## 7. Truth-late persistent-episode and safety readout",
        "## 8. Stage 0 gates, generated artifacts, and metrics",
    ):
        assert heading in train_source
    parameters = inspect.signature(train.freeze_target_free_well).parameters
    assert not {
        "truth",
        "error",
        "episodes",
        "fold",
        "role",
        "hidden_like_role",
    }.intersection(parameters)
