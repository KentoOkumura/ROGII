from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.util
import inspect
import json
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
    / "exp445_tvt_to_u_coordinate_parity_exact_hmm"
)
TRAIN_SOURCE = (
    EXP_DIR
    / "exp445_tvt_to_u_coordinate_parity_exact_hmm_compact_selfcontained_train.py"
)
INFERENCE_SOURCE = (
    EXP_DIR
    / "exp445_tvt_to_u_coordinate_parity_exact_hmm_compact_selfcontained_inference.py"
)
CONFIG_PATH = EXP_DIR / "config.yaml"
MANIFEST_PATH = (
    ROOT
    / "experiments"
    / "exp411_predictive_filtered_rate_innovation_destick"
    / "assets"
    / "stage0_fixed32_manifest.csv"
)
CANONICAL_TRAIN = (
    EXP_DIR / "exp445_tvt_to_u_coordinate_parity_exact_hmm_train.ipynb"
)
CANONICAL_INFERENCE = (
    EXP_DIR / "exp445_tvt_to_u_coordinate_parity_exact_hmm_inference.ipynb"
)
PARENT_COMPACT = (
    ROOT
    / "experiments"
    / "exp438_u_state_fixed_lattice_exact_hmm"
    / "exp438_u_state_fixed_lattice_exact_hmm_compact_selfcontained_train.py"
)
PARENT_HMM_SOURCE = (
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
    return load_module(TRAIN_SOURCE, "exp445_train_test")


@pytest.fixture(scope="module")
def inference():
    return load_module(INFERENCE_SOURCE, "exp445_inference_test")


@pytest.fixture(scope="module")
def parent_hmm():
    return load_module(PARENT_HMM_SOURCE, "exp445_exp209_parent_test")


@pytest.fixture(scope="module")
def config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())


def mini_hmm(config: dict) -> dict:
    hmm = deepcopy(config["model"]["fixed_from_exp209"])
    hmm["n_rates"] = 3
    hmm["rate_span"] = 0.02
    hmm["band_pad_ft"] = 1.05
    return hmm


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_execution_contract_authorizes_fixed32_stage0_only(train, config):
    counts = train.validate_execution_contract(
        config,
        require_run_authorization=False,
    )
    assert counts == {
        "coordinate_candidates": 1,
        "manifest_wells": 32,
        "candidate_hmm_well_runs": 32,
        "paired_parent_hmm_well_runs": 32,
        "total_hmm_well_runs": 64,
        "reporting_folds": 0,
        "lightgbm_configs": 0,
        "trained_ml_folds": 0,
        "boosters": 0,
        "fitted_models": 0,
        "pf_runs": 0,
        "beam_runs": 0,
        "gpu_runs": 0,
    }
    assert config["experiment"]["status"] != "implementation_complete_not_run"
    assert config["design"]["implementation_authorized"] is True
    assert config["design"]["canonical_notebook_adoption_authorized"] is True
    assert config["design"]["kaggle_package_authorized"] is True
    assert config["design"]["kaggle_run_authorized"] is True
    assert config["execution"]["selected_stage"] == "stage_0_fixed32"
    authorized = deepcopy(config)
    authorized["runtime"]["run_approved"] = True
    authorized["execution"]["run_hmm"] = True
    train.validate_execution_contract(
        authorized,
        require_run_authorization=True,
    )
    locked = deepcopy(config)
    locked["runtime"]["run_approved"] = False
    locked["execution"]["run_hmm"] = False
    with pytest.raises(
        RuntimeError,
        match="does not authorize Kaggle execution",
    ):
        train.validate_execution_contract(
            locked,
            require_run_authorization=True,
        )


def test_scientific_contract_pins_row_shifted_u_not_fixed_u(train, config):
    contract = train.validate_scientific_contract(config)
    coordinate = contract["coordinate"]
    transition = contract["transition"]
    assert coordinate["candidate_u_grid_formula"] == "U_t_j=P_j+Z_t"
    assert coordinate["candidate_tvt_view_formula"] == "U_t_j-Z_t=P_j"
    assert coordinate["candidate_grid_is_fixed_absolute_u"] is False
    assert (
        transition["candidate_index_mean_formula"]
        == "r_current*delta_MD-delta_Z"
    )
    assert (
        transition["candidate_physical_u_edge_formula"]
        == "(P_k-P_j)+delta_Z"
    )

    broken = deepcopy(config)
    broken["model"]["coordinate"]["candidate_grid_is_fixed_absolute_u"] = True
    with pytest.raises(ValueError, match="coordinate contract"):
        train.validate_scientific_contract(broken)


@pytest.mark.parametrize("variable_z", [False, True])
def test_coordinate_emission_prior_and_index_mean_are_exact(
    train,
    config,
    variable_z,
):
    horizontal, typewell = train.synthetic_inputs(
        variable_z=variable_z,
        rows=3,
    )
    prepared = train.prepare_paired_inputs(
        horizontal,
        typewell,
        mini_hmm(config),
    )
    contract = train.coordinate_contract_from_prepared(prepared)
    assert (
        contract["coordinate_tvt_equals_u_minus_z_max_abs_ft"]
        <= 1.0e-12
    )
    assert contract["transition_index_mean_max_abs_ft"] <= 1.0e-12
    assert contract["emission_max_abs"] <= 1.0e-12
    assert contract["initial_prior_max_abs"] <= 1.0e-12
    parent = prepared["parent"]
    candidate = prepared["candidate"]
    assert not np.shares_memory(
        parent["emission_ll"],
        candidate["emission_ll"],
    )
    assert not np.shares_memory(
        parent["initial_position_log_prior"],
        candidate["initial_position_log_prior"],
    )


@pytest.mark.parametrize(
    ("rate", "delta_md", "delta_z"),
    [
        (0.017, 10.0, 0.13),
        (-0.025, 7.5, -0.41),
        (0.0, 15.0, 0.0),
        (0.083, 3.0, 1.17),
    ],
)
def test_candidate_physical_edge_kernel_equals_parent_index_kernel(
    train,
    rate,
    delta_md,
    delta_z,
):
    parent_offsets, parent_weights = (
        train.parent_position_kernel_probabilities(
            rate,
            delta_md,
            delta_z,
            0.35,
            0.02,
        )
    )
    candidate_offsets, candidate_weights = (
        train.candidate_u_position_kernel_probabilities(
            rate,
            delta_md,
            delta_z,
            0.35,
            0.02,
        )
    )
    np.testing.assert_array_equal(parent_offsets, candidate_offsets)
    np.testing.assert_allclose(
        parent_weights,
        candidate_weights,
        rtol=0.0,
        atol=1.0e-12,
    )
    parent_residual = (
        parent_offsets.astype(np.float64) * 0.35
        - (rate * delta_md - delta_z)
    )
    candidate_residual = (
        candidate_offsets.astype(np.float64) * 0.35
        + delta_z
        - rate * delta_md
    )
    np.testing.assert_allclose(
        parent_residual,
        candidate_residual,
        rtol=0.0,
        atol=1.0e-12,
    )


def test_independent_parent_and_candidate_hmms_have_parity(train, config):
    horizontal, typewell = train.synthetic_inputs(
        variable_z=True,
        rows=3,
    )
    hmm = mini_hmm(config)
    prepared = train.prepare_paired_inputs(horizontal, typewell, hmm)
    result = train.run_paired_hmms(prepared, hmm)
    assert result["position_posterior_max_abs"] <= 1.0e-8
    assert result["rate_posterior_max_abs"] <= 1.0e-8
    assert result["log_likelihood_abs"] <= 1.0e-6
    assert result["tvt_mean_max_abs_ft"] <= 1.0e-6
    assert result["tvt_std_max_abs_ft"] <= 1.0e-6
    assert result["candidate_u_minus_z_readout_max_abs_ft"] <= 1.0e-6
    assert result["finite_coverage"] == 1.0
    assert result["parent_posterior_position"] is not (
        result["candidate_posterior_position"]
    )
    assert result["parent_posterior_rate"] is not (
        result["candidate_posterior_rate"]
    )


def test_parent_coordinate_kernel_matches_exp209_exactly(train, parent_hmm):
    emission = np.asarray(
        [[-1.2, -0.1, -2.7], [-2.0, -0.4, -1.1]],
        dtype=np.float32,
    )
    dm = np.asarray([1.0, 1.3], dtype=np.float64)
    dz = np.asarray([0.08, -0.03], dtype=np.float64)
    step = 0.35
    rates = np.asarray([-0.08, 0.0, 0.08], dtype=np.float64)
    start_p = 1.1
    start_sig = 0.75
    r0 = 0.01
    r0_sig = 0.08
    sig_r = 0.02
    sig_p = 0.12
    emission_lambda = 1.0
    momentum = 0.998
    initial_prior = train.exp209_initial_position_log_prior(
        emission.shape[1],
        start_p,
        step,
        start_sig,
    )
    observed_position, _, observed_loglik, _ = train._hmm2_fb_coordinate(
        0,
        emission,
        dm,
        dz,
        step,
        rates,
        sig_r,
        sig_p,
        initial_prior,
        r0,
        r0_sig,
        emission_lambda,
        momentum,
    )
    expected_position, expected_loglik = parent_hmm._hmm2_fb(
        emission,
        dm,
        dz,
        step,
        rates,
        sig_r,
        sig_p,
        start_p,
        start_sig,
        r0,
        r0_sig,
        emission_lambda,
        momentum,
    )
    np.testing.assert_array_equal(observed_position, expected_position)
    assert observed_loglik == expected_loglik


def test_tiny_hmm_matches_brute_force_for_both_coordinates(train, config):
    contract = train.brute_force_small_reference_contract(
        config["model"]["fixed_from_exp209"]
    )
    assert contract["pass"] is True
    assert contract["maximum_abs"] <= 1.0e-6
    assert contract["paired_position_posterior_max_abs"] <= 1.0e-8
    assert contract["paired_rate_posterior_max_abs"] <= 1.0e-8
    assert contract["paired_log_likelihood_abs"] <= 1.0e-6


def test_fixed32_manifest_is_sha_pinned_and_target_free(train, config):
    assert MANIFEST_PATH.is_file()
    assert sha256(MANIFEST_PATH) == (
        config["data"]["fixed32_manifest"]["expected_sha256"]
    )
    ledger = train.LeakageLedger()
    frame, report = train.load_fixed32_target_free_scope(config, ledger)
    assert report["sha256"] == sha256(MANIFEST_PATH)
    assert list(frame.columns) == ["well", "prefix_rows", "suffix_rows"]
    assert len(frame) == 32
    assert frame["well"].nunique() == 32
    assert ledger.suffix_truth_reads == 0
    assert ledger.fold_reads == 0
    assert ledger.role_reads == 0
    assert ledger.episode_reads == 0
    assert ledger.error_reads == 0


def test_deterministic_gzip_readback_uses_decompressed_content_sha(
    train,
    tmp_path,
):
    frame = pd.DataFrame(
        {
            "well": ["a", "b"],
            "value": [0.12345678901234567, -4.25],
        }
    )
    path = tmp_path / "ledger.csv.gz"
    first = train.write_deterministic_gzip_csv(path, frame)
    first_bytes = path.read_bytes()
    second = train.write_deterministic_gzip_csv(path, frame)
    assert path.read_bytes() == first_bytes
    assert first["raw_sha256"] == second["raw_sha256"]
    assert first["decompressed_sha256"] == second["decompressed_sha256"]
    assert first["readback_match"] is True
    assert first["decompressed_sha256"] == train.sha256_decompressed_csv(path)


def test_compact_train_is_not_a_thin_helper_entrypoint(train):
    source = TRAIN_SOURCE.read_text()
    assert "__file__" not in source
    assert "from settings import" not in source
    assert "def assemble_parent_tvt_inputs(" in source
    assert "def assemble_candidate_u_inputs(" in source
    assert "def _hmm2_fb_coordinate(" in source
    assert "def exhaustive_small_path_reference(" in source
    assert "def run_stage0(" in source
    assert 'os.environ["NUMBA_NUM_THREADS"]' not in source
    headings = [
        line
        for line in source.splitlines()
        if line.startswith("# ## ")
    ]
    assert len(headings) == 10
    assert len(source.splitlines()) >= 2_000
    assert len(PARENT_COMPACT.read_text().splitlines()) >= 2_500
    loader_source = inspect.getsource(train.load_target_free_well)
    assert 'usecols=["MD", "Z", "GR", "TVT_input"]' in loader_source
    assert "suffix_truth" not in loader_source


def test_inference_is_permanently_fail_closed(inference, config):
    contract = inference.validate_inference_disabled(config)
    assert contract["implementation_authorized"] is True
    assert contract["canonical_notebook_adoption_authorized"] is True
    assert contract["kaggle_package_authorized"] is True
    assert contract["kaggle_run_authorized"] is True
    assert contract["inference_authorized"] is False
    assert contract["submission_authorized"] is False
    assert contract["create_submission"] is False
    with pytest.raises(RuntimeError, match="no inference stage"):
        inference.run_inference(config)


def test_canonical_notebooks_are_self_contained_train_and_inference_guard():
    train_notebook = json.loads(CANONICAL_TRAIN.read_text())
    inference_notebook = json.loads(CANONICAL_INFERENCE.read_text())
    train_source = "\n".join(
        "".join(cell["source"]) for cell in train_notebook["cells"]
    )
    inference_source = "\n".join(
        "".join(cell["source"]) for cell in inference_notebook["cells"]
    )
    assert len(train_notebook["cells"]) >= 20
    assert len(inference_notebook["cells"]) >= 6
    assert "Metrics scaffold" not in train_source
    assert "from settings import" not in train_source
    assert "from settings import" not in inference_source
    assert "assemble_candidate_u_inputs" in train_source
    assert "exp445 has no inference stage" in inference_source


def test_gate_evaluator_rejects_any_forbidden_read(train, config):
    parity = pd.DataFrame(
        [
            {
                "coordinate_tvt_equals_u_minus_z_max_abs_ft": 0.0,
                "physical_edge_residual_identity_max_abs_ft": 0.0,
                "emission_max_abs": 0.0,
                "initial_prior_max_abs": 0.0,
                "position_posterior_max_abs": 0.0,
                "rate_posterior_max_abs": 0.0,
                "log_likelihood_abs": 0.0,
                "tvt_mean_max_abs_ft": 0.0,
                "tvt_std_max_abs_ft": 0.0,
                "candidate_u_minus_z_readout_max_abs_ft": 0.0,
                "finite_coverage": 1.0,
            }
        ]
        * 32
    )
    transition = pd.DataFrame(
        [
            {
                "rate_kernel_max_abs": 0.0,
                "position_kernel_max_abs": 0.0,
            }
        ]
        * 32
    )
    synthetic_branch = {
        "coordinate": {
            "coordinate_tvt_equals_u_minus_z_max_abs_ft": 0.0,
            "emission_max_abs": 0.0,
            "initial_prior_max_abs": 0.0,
        },
        "transition": {
            "physical_edge_residual_identity_max_abs_ft": 0.0,
            "rate_kernel_max_abs": 0.0,
            "position_kernel_max_abs": 0.0,
        },
    }
    synthetic = {
        "variable_z": synthetic_branch,
        "constant_z": deepcopy(synthetic_branch),
        "brute_force": {"maximum_abs": 0.0},
    }
    leakage = {
        "suffix_truth_reads": 1,
        "fold_reads": 0,
        "role_reads": 0,
        "episode_reads": 0,
        "error_reads": 0,
    }
    result = train.evaluate_technical_gates(
        config=config,
        synthetic=synthetic,
        parity_ledger=parity,
        transition_ledger=transition,
        leakage=leakage,
        artifacts={"ledger": {"readback_match": True}},
    )
    assert result["checks"]["truth_free"] is False
    assert result["all_pass"] is False
    assert result["decision"] == "technical_parity_failed"
