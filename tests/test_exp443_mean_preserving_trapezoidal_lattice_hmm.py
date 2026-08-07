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
    numba_stub.set_num_threads = lambda threads: None
    numba_stub.__version__ = "test-stub"
    sys.modules["numba"] = numba_stub

ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = (
    ROOT
    / "experiments"
    / "exp443_mean_preserving_trapezoidal_lattice_hmm"
)
TRAIN_SOURCE = (
    EXP_DIR
    / "exp443_mean_preserving_trapezoidal_lattice_hmm_compact_selfcontained_train.py"
)
INFERENCE_SOURCE = (
    EXP_DIR
    / "exp443_mean_preserving_trapezoidal_lattice_hmm_compact_selfcontained_inference.py"
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
    return load_module(TRAIN_SOURCE, "exp443_train_test")


@pytest.fixture(scope="module")
def inference():
    return load_module(INFERENCE_SOURCE, "exp443_inference_test")


@pytest.fixture(scope="module")
def config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())


def synthetic_inputs(rows: int = 6) -> tuple[pd.DataFrame, pd.DataFrame]:
    prefix_rows = 8
    total = prefix_rows + rows
    md = np.arange(total, dtype=np.float64) * 10.0
    z = np.full(total, 8_000.0, dtype=np.float64)
    visible_tvt = 12_000.0 + 0.02 * md
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


def test_stage0_execution_contract_is_terminally_locked_after_run(
    train,
    config,
):
    counts = train.validate_execution_contract(
        config,
        require_run_authorization=False,
    )
    assert counts == {
        "scientific_variants": 1,
        "stage0_candidate_hmm_well_runs": 32,
        "stage1_candidate_hmm_well_runs": 773,
        "parent_control_hmm_well_runs": 0,
        "lightgbm_configs": 0,
        "trained_ml_folds": 0,
        "boosters": 0,
        "fitted_models": 0,
        "pf_runs": 0,
        "beam_runs": 0,
        "gpu_runs": 0,
    }
    assert config["experiment"]["status"] == "stage0_fail_closed_v1"
    assert config["execution"]["implementation_authorized"] is True
    assert config["execution"]["canonical_notebook_adoption_authorized"] is True
    assert config["execution"]["kaggle_package_authorized"] is True
    assert config["design"]["stage0_run_authorized"] is True
    assert config["execution"]["stage0_run_authorized"] is False
    assert config["execution"]["run_hmm"] is False
    assert config["execution"]["create_prediction"] is False
    assert config["implementation"]["stage0_completed"] is True
    with pytest.raises(
        RuntimeError,
        match="does not authorize Stage 0 execution",
    ):
        train.validate_execution_contract(
            config,
            require_run_authorization=True,
        )

    broken = deepcopy(config)
    broken["execution"]["stage1_run_authorized"] = True
    with pytest.raises(ValueError, match="Stage 1"):
        train.validate_execution_contract(
            broken,
            require_run_authorization=False,
        )


def test_scientific_contract_pins_trapezoidal_joint_edge(train, config):
    contract = train.validate_scientific_contract(config)
    assert contract["candidate_state"] == ["tvt_position", "u_rate"]
    assert contract["candidate_joint_transition"][
        "conditional_position_mean_formula"
    ] == "0.5*(r_source+r_destination)*delta_MD-delta_Z"
    assert contract["candidate_joint_transition"][
        "rate_marginal"
    ] == "exact_exp209_adjacent_three_state_kernel"
    assert contract["candidate_joint_transition"]["support_cells"] == 5
    assert contract["candidate_joint_transition"][
        "effective_variance_formula"
    ] == "max(parent_target_variance,minimum_lattice_variance)"

    broken = deepcopy(config)
    broken["model"]["fixed_from_exp209"]["sig_p"] = 0.03
    with pytest.raises(ValueError, match="exp209 HMM contract changed"):
        train.validate_scientific_contract(broken)


def test_preparation_preserves_exp209_fixed_tvt_emission(train, config):
    horizontal, typewell = synthetic_inputs()
    hmm = config["model"]["fixed_from_exp209"]
    prepared = train.prepare_hmm_inputs(horizontal, typewell, hmm)
    assert "tvt_grid" in prepared
    assert "u_grid" not in prepared
    expected_gr = np.interp(
        prepared["tvt_grid"],
        prepared["typewell_tvt"],
        prepared["typewell_gr"],
    )
    expected = -0.5 * np.minimum(
        (
            (prepared["gr"][:, None] - expected_gr[None, :])
            / prepared["prefix_sigma"]
        )
        ** 2,
        600.0,
    )
    np.testing.assert_allclose(
        prepared["emission_ll_exact"],
        expected,
        rtol=0.0,
        atol=0.0,
    )
    assert (
        train.input_contract_from_prepared(prepared)[
            "emission_identity_max_abs"
        ]
        == 0.0
    )


def test_maximum_entropy_projection_preserves_probability_and_moments(
    train,
    config,
):
    projection = config["model"]["candidate_joint_transition"]
    solver = projection["solver"]
    mean_shift = 0.0
    step = 0.35
    variance = 0.1225**2
    offsets, weights, count, feasible, iterations = (
        train.moment_preserving_projection(
            mean_shift,
            step,
            variance,
            solver["maximum_iterations"],
            solver["moment_tolerance"],
            solver["feasibility_tolerance"],
            solver["damping_min"],
        )
    )
    assert feasible is True
    assert count == 5
    assert iterations <= solver["maximum_iterations"]
    total, mean, observed_variance = train.projection_moments(
        mean_shift,
        step,
        offsets,
        weights,
        count,
    )
    assert total == pytest.approx(1.0, abs=1.0e-12)
    assert mean == pytest.approx(mean_shift, abs=1.0e-10)
    assert observed_variance == pytest.approx(variance, abs=1.0e-10)
    assert np.all(weights[:count] >= 0.0)


def test_exp439_failure_edge_is_feasible_with_explicit_variance_floor(
    train,
    config,
):
    projection = config["model"]["candidate_joint_transition"]
    solver = projection["solver"]
    parent_variance = 0.1225**2
    minimum, effective, inflation = train.effective_lattice_variance(
        0.11,
        0.35,
        parent_variance,
    )
    assert minimum == pytest.approx(0.0264, abs=1.0e-14)
    assert effective == pytest.approx(0.0264, abs=1.0e-14)
    assert inflation == pytest.approx(0.0264 - parent_variance, abs=1.0e-14)
    offsets, weights, count, feasible, _ = (
        train.moment_preserving_projection(
            0.11,
            0.35,
            effective,
            solver["maximum_iterations"],
            solver["moment_tolerance"],
            solver["feasibility_tolerance"],
            solver["damping_min"],
        )
    )
    assert feasible is True
    assert count == 5
    total, mean, variance = train.projection_moments(
        0.11,
        0.35,
        offsets,
        weights,
        count,
    )
    assert total == pytest.approx(1.0, abs=1.0e-12)
    assert mean == pytest.approx(0.11, abs=1.0e-10)
    assert variance == pytest.approx(effective, abs=1.0e-10)
    assert np.count_nonzero(weights[:count] < 0.0) == 0
    contract = train.exp439_failure_edge_variance_floor_contract(projection)
    assert contract["pass"] is True
    assert contract["support_cells"] == 5
    assert contract["minimum_lattice_variance_ft2"] == pytest.approx(
        0.0264,
        abs=1.0e-14,
    )


def test_joint_table_preserves_rate_marginal_mean_variance_and_covariance(
    train,
    config,
):
    hmm = config["model"]["fixed_from_exp209"]
    projection = config["model"]["candidate_joint_transition"]
    dm = np.asarray([1.0, 1.0], dtype=np.float64)
    dz = np.asarray([-0.11, -0.175], dtype=np.float64)
    rates = np.asarray([-0.70, 0.0, 0.70], dtype=np.float64)
    table = train.build_joint_edge_table(
        dm=dm,
        dz=dz,
        rates=rates,
        step=hmm["position_grid_step_ft"],
        sig_r=hmm["sig_r"],
        sig_p=hmm["sig_p"],
        momentum=hmm["momentum"],
        projection=projection,
    )
    audit = train.joint_edge_moment_audit(
        dm=dm,
        dz=dz,
        rates=rates,
        step=hmm["position_grid_step_ft"],
        sig_r=hmm["sig_r"],
        sig_p=hmm["sig_p"],
        momentum=hmm["momentum"],
        table=table,
        row_idx=np.arange(2),
    )
    assert audit["rate_marginal_max_abs_error"].max() <= 1.0e-12
    assert audit["legal_edge_weight_sum_max_error"].max() <= 1.0e-12
    assert audit["conditional_mean_max_abs_error_ft"].max() <= 1.0e-10
    assert audit["effective_variance_max_abs_error_ft2"].max() <= 1.0e-10
    assert audit["negative_weight_count"].sum() == 0
    assert audit["fixed_support_violation_edges"].sum() == 0
    assert audit["support_5_edges"].sum() == audit["legal_edges"].sum()
    assert audit["variance_inflation_max_ft2"].max() > 0.0
    assert (
        audit["source_row_joint_covariance_max_abs_error"].max()
        <= 1.0e-10
    )
    assert audit["forward_backward_joint_table_identity"].all()

    parent_rate = train.rate_kernel_probabilities(
        rates,
        1.0,
        hmm["sig_r"],
        hmm["momentum"],
    )
    np.testing.assert_array_equal(table["rate_probability"][0], parent_rate)
    assert table["counts"][0, 0, 0] == 0
    assert parent_rate[0, 0] > 0.0


def test_joint_forward_backward_matches_exhaustive_paths(train, config):
    contract = train.brute_force_joint_reference_contract(
        config["model"]["fixed_from_exp209"],
        config["model"]["candidate_joint_transition"],
    )
    assert contract["pass"] is True
    assert contract["position_posterior_max_abs"] <= 1.0e-6
    assert contract["rate_posterior_max_abs"] <= 1.0e-6
    assert contract["prediction_max_abs_ft"] <= 1.0e-6
    assert contract["posterior_normalization_max_error"] <= 1.0e-6


def test_joint_edge_table_sha_is_deterministic(train, config):
    hmm = config["model"]["fixed_from_exp209"]
    kwargs = {
        "dm": np.ones(2, dtype=np.float64),
        "dz": np.zeros(2, dtype=np.float64),
        "rates": np.asarray([-0.70, 0.0, 0.70], dtype=np.float64),
        "step": hmm["position_grid_step_ft"],
        "sig_r": hmm["sig_r"],
        "sig_p": hmm["sig_p"],
        "momentum": hmm["momentum"],
        "projection": config["model"]["candidate_joint_transition"],
    }
    first = train.build_joint_edge_table(**kwargs)
    second = train.build_joint_edge_table(**kwargs)
    assert first["sha256"] == second["sha256"]
    np.testing.assert_array_equal(first["weights"], second["weights"])


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
        config["data"]["exp209_saved_control"][
            "expected_decompressed_sha256"
        ]
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
    for heading in (
        "## 1. Imports and immutable execution contract",
        "## 3. Fixed32 manifest, saved parent, and target-free raw inputs",
        "## 4. Exact exp209 input preparation",
        "## 5. Mean-preserving lattice-floor joint-edge implementation",
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
