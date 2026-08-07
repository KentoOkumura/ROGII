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
    numba_stub.__spec__ = importlib.machinery.ModuleSpec(
        "numba",
        loader=None,
    )

    def _njit(*args, **kwargs):
        del kwargs
        if args and callable(args[0]):
            return args[0]

        def decorator(function):
            return function

        return decorator

    numba_stub.njit = _njit
    numba_stub.prange = range
    numba_stub.get_num_threads = lambda: 1
    numba_stub.set_num_threads = lambda threads: None
    numba_stub.__version__ = "test-stub"
    sys.modules["numba"] = numba_stub

ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = (
    ROOT
    / "experiments"
    / "exp442_symmetric_broad_jump_rate_transition_hmm"
)
TRAIN_SOURCE = (
    EXP_DIR
    / "exp442_symmetric_broad_jump_rate_transition_hmm_"
    "compact_selfcontained_train.py"
)
INFERENCE_SOURCE = (
    EXP_DIR
    / "exp442_symmetric_broad_jump_rate_transition_hmm_"
    "compact_selfcontained_inference.py"
)
CONFIG_PATH = EXP_DIR / "config.yaml"
EXP209_REFERENCE_SOURCE = (
    ROOT
    / "experiments"
    / "exp209_exp072_exp205_joint_exact_parity_fast_cache_generation"
    / "exact_hmm_smoother.py"
)
MANIFEST_PATH = (
    ROOT
    / "experiments"
    / "exp411_predictive_filtered_rate_innovation_destick"
    / "assets"
    / "stage0_fixed32_manifest.csv"
)
CANONICAL_TRAIN = (
    EXP_DIR
    / "exp442_symmetric_broad_jump_rate_transition_hmm_train.ipynb"
)
CANONICAL_INFERENCE = (
    EXP_DIR
    / "exp442_symmetric_broad_jump_rate_transition_hmm_inference.ipynb"
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
    return load_module(TRAIN_SOURCE, "exp442_train_test")


@pytest.fixture(scope="module")
def inference():
    return load_module(INFERENCE_SOURCE, "exp442_inference_test")


@pytest.fixture(scope="module")
def exp209_reference():
    return load_module(
        EXP209_REFERENCE_SOURCE,
        "exp442_exp209_reference_test",
    )


@pytest.fixture(scope="module")
def config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())


def synthetic_prepared(
    fixed: dict,
    *,
    rows: int = 5,
    positions: int = 23,
) -> dict:
    grid = 11_900.0 + np.arange(positions, dtype=np.float64) * float(
        fixed["position_grid_step_ft"]
    )
    rates = np.linspace(
        -float(fixed["rate_span"]),
        float(fixed["rate_span"]),
        int(fixed["n_rates"]),
        dtype=np.float64,
    )
    x = np.linspace(-1.0, 1.0, positions)
    emission = np.vstack(
        [
            -0.5
            * (
                (x - 0.2 * np.sin(row / 2.0))
                / 0.38
            )
            ** 2
            for row in range(rows)
        ]
    ).astype(np.float32)
    return {
        "emission_ll": emission,
        "dm": np.linspace(6.0, 18.0, rows, dtype=np.float64),
        "dz": np.linspace(-0.3, 0.4, rows, dtype=np.float64),
        "grid": grid,
        "rates": rates,
        "start_p": 11.0,
        "r0": 0.0,
        "eval_index": np.arange(100, 100 + rows, dtype=np.int64),
    }


def test_stage0_execution_contract_is_authorized_and_later_stages_fail_closed(
    train,
    config,
):
    observed = train.validate_execution_contract(
        config,
        require_run_authorization=False,
    )
    assert observed == {
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
    assert config["experiment"]["status"] == "stage0_fail_closed"
    assert config["execution"]["implementation_authorized"] is True
    assert (
        config["execution"]["canonical_notebook_adoption_authorized"]
        is True
    )
    assert config["execution"]["kaggle_package_authorized"] is True
    assert config["execution"]["stage0_run_authorized"] is True
    assert config["execution"]["run_hmm"] is True
    assert config["execution"]["create_prediction"] is True
    assert (
        train.validate_execution_contract(
            config,
            require_run_authorization=True,
        )
        == observed
    )

    broken = deepcopy(config)
    broken["execution"]["stage1_run_authorized"] = True
    with pytest.raises(ValueError, match="Stage 1"):
        train.validate_execution_contract(
            broken,
            require_run_authorization=False,
        )


def test_scientific_contract_pins_single_broad_candidate(
    train,
    config,
):
    contract = train.validate_scientific_contract(config)
    candidate = contract["candidate_rate_transition"]
    assert candidate["jump_weight"] == 0.01
    assert candidate["broad_sigma_rate"] == 0.02
    assert candidate["symmetry"] == "target_free_two_sided"
    assert candidate["future_rate_horizon_rows"] == 32
    assert contract["active_scientific_variants"] == [
        "symmetric_broad_jump_w001_s002"
    ]

    broken = deepcopy(config)
    broken["model"]["candidate_rate_transition"]["jump_weight"] = 0.02
    with pytest.raises(ValueError, match="broad-jump contract"):
        train.validate_scientific_contract(broken)


def test_local_broad_and_mixture_kernel_contracts(train, config):
    checks = train.synthetic_kernel_contract(
        config["model"]["fixed_from_exp209"],
        config["model"]["candidate_rate_transition"],
    )
    assert checks[
        "local_branch_parent_parity_max_abs_error"
    ] <= 1.0e-12
    assert checks[
        "mixture_decomposition_max_abs_error"
    ] <= 1.0e-12
    assert checks[
        "broad_in_support_mass_max_abs_error"
    ] <= 1.0e-12
    assert checks[
        "centered_broad_symmetry_max_abs_error"
    ] <= 1.0e-12
    assert checks[
        "brute_force_branch_responsibility_max_abs_error"
    ] <= 1.0e-12


def test_broad_boundary_mass_is_discarded_without_renormalization(
    train,
    config,
):
    fixed = config["model"]["fixed_from_exp209"]
    candidate = config["model"]["candidate_rate_transition"]
    rates = np.linspace(
        -fixed["rate_span"],
        fixed["rate_span"],
        fixed["n_rates"],
    )
    broad, analytic_mass = train.broad_rate_kernel(
        rates,
        20.0,
        fixed["momentum"],
        candidate["broad_sigma_rate"],
    )
    np.testing.assert_allclose(
        broad.sum(axis=1),
        analytic_mass,
        rtol=0.0,
        atol=1.0e-12,
    )
    assert broad[0].sum() < 1.0
    assert broad[-1].sum() < 1.0
    assert broad[len(rates) // 2].sum() > broad[0].sum()


def test_branch_responsibility_matches_explicit_edge_enumeration(
    train,
    config,
):
    fixed = config["model"]["fixed_from_exp209"]
    candidate = config["model"]["candidate_rate_transition"]
    rates = np.linspace(
        -fixed["rate_span"],
        fixed["rate_span"],
        fixed["n_rates"],
    )
    local, broad, _, _ = train.mixed_rate_kernel(
        rates,
        12.0,
        fixed["sig_r"],
        fixed["momentum"],
        candidate["jump_weight"],
        candidate["broad_sigma_rate"],
    )
    rng = np.random.default_rng(442)
    alpha = rng.uniform(0.1, 1.0, size=(4, len(rates)))
    alpha /= alpha.sum()
    beta = rng.uniform(0.1, 1.0, size=(4, len(rates)))
    observed = train.branch_responsibility_from_messages(
        alpha,
        beta,
        local,
        broad,
        rates,
        candidate["jump_weight"],
    )
    denominator = 0.0
    broad_total = 0.0
    nonadjacent_total = 0.0
    signed_total = 0.0
    for position in range(alpha.shape[0]):
        for source in range(len(rates)):
            for destination in range(len(rates)):
                common = (
                    alpha[position, source]
                    * beta[position, destination]
                )
                local_edge = (
                    (1.0 - candidate["jump_weight"])
                    * local[source, destination]
                )
                broad_edge = (
                    candidate["jump_weight"]
                    * broad[source, destination]
                )
                denominator += common * (local_edge + broad_edge)
                broad_total += common * broad_edge
                if abs(destination - source) > 1:
                    nonadjacent_total += common * broad_edge
                    signed_total += (
                        common
                        * broad_edge
                        * (rates[destination] - rates[source])
                    )
    expected = np.asarray(
        [
            broad_total / denominator,
            nonadjacent_total / denominator,
            signed_total / nonadjacent_total,
        ]
    )
    np.testing.assert_allclose(
        np.asarray(observed[:3]),
        expected,
        rtol=0.0,
        atol=1.0e-12,
    )


def test_jump_weight_zero_matches_independent_exp209_reference(
    train,
    exp209_reference,
    config,
):
    fixed = config["model"]["fixed_from_exp209"]
    candidate = config["model"]["candidate_rate_transition"]
    prepared = synthetic_prepared(fixed)
    decoded = train.run_symmetric_broad_jump_hmm(
        prepared,
        fixed,
        candidate,
        jump_weight_override=0.0,
    )
    reference_posterior, reference_loglik = exp209_reference._hmm2_fb(
        prepared["emission_ll"],
        prepared["dm"],
        prepared["dz"],
        float(fixed["position_grid_step_ft"]),
        prepared["rates"],
        float(fixed["sig_r"]),
        float(fixed["sig_p"]),
        float(prepared["start_p"]),
        float(fixed["start_sigma_ft"]),
        float(prepared["r0"]),
        float(fixed["initial_rate_sigma"]),
        float(fixed["emission_lambda"]),
        float(fixed["momentum"]),
    )
    reference_mean = reference_posterior @ prepared["grid"]
    np.testing.assert_allclose(
        decoded["posterior_mean"],
        reference_mean,
        rtol=0.0,
        atol=2.0e-3,
    )
    assert decoded["log_likelihood"] == pytest.approx(
        reference_loglik,
        abs=2.0e-3,
    )
    assert np.count_nonzero(decoded["branch_responsibility"]) == 0
    assert np.count_nonzero(decoded["nonadjacent_edge_mass"]) == 0


def test_broad_candidate_creates_finite_nonadjacent_posterior_mass(
    train,
    config,
):
    fixed = config["model"]["fixed_from_exp209"]
    candidate = config["model"]["candidate_rate_transition"]
    prepared = synthetic_prepared(fixed, rows=4, positions=19)
    decoded = train.run_symmetric_broad_jump_hmm(
        prepared,
        fixed,
        candidate,
    )
    assert np.isfinite(decoded["posterior_mean"]).all()
    assert np.isfinite(decoded["posterior_std"]).all()
    assert np.all(decoded["branch_responsibility"][1:] > 0.0)
    assert np.all(decoded["nonadjacent_edge_mass"][1:] > 0.0)
    assert decoded["maximum_normalization_error"] <= 1.0e-12
    assert len(decoded["transition_kernel_sha256"]) == 64
    assert len(decoded["responsibility_sha256"]) == 64


def test_leakage_ledger_blocks_identity_truth_episode_and_cause_until_freeze(
    train,
):
    ledger = train.LeakageLedger(expected_wells=2)
    ledger.freeze(
        "a",
        schedule_sha256="transition-a",
        prediction_sha256="prediction-a",
        diagnostic_sha256="diagnostic-a",
    )
    with pytest.raises(RuntimeError, match="role/fold"):
        ledger.record_role_fold_late(1)
    with pytest.raises(RuntimeError, match="truth"):
        ledger.record_truth_late(2)
    with pytest.raises(RuntimeError, match="episodes"):
        ledger.record_episode_late(3)
    with pytest.raises(RuntimeError, match="causes"):
        ledger.record_cause_late(4)
    assert ledger.forbidden_reads_before_all_freeze == 10
    ledger.freeze(
        "b",
        schedule_sha256="transition-b",
        prediction_sha256="prediction-b",
        diagnostic_sha256="diagnostic-b",
    )
    ledger.record_role_fold_late(2)
    ledger.record_truth_late(5)
    ledger.record_episode_late(1)
    ledger.record_cause_late(1)
    assert ledger.all_frozen


def test_fixed32_manifest_is_sha_pinned_balanced_and_unique(config):
    observed = hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest()
    assert observed == (
        config["data"]["fixed32_manifest"]["expected_sha256"]
    )
    manifest = pd.read_csv(MANIFEST_PATH, dtype={"well": str})
    assert len(manifest) == 32
    assert manifest["well"].nunique() == 32
    assert int(manifest["suffix_rows"].sum()) == 156_088
    assert manifest["role"].value_counts().to_dict() == {
        "persistent": 16,
        "control": 16,
    }
    assert manifest["fold"].nunique() == 5


def test_inference_remains_fail_closed(inference, config):
    contract = inference.validate_inference_disabled(config)
    assert contract["implementation_authorized"] is True
    assert contract["canonical_notebook_adoption_authorized"] is True
    assert contract["kaggle_package_authorized"] is True
    assert contract["stage0_run_authorized"] is True
    assert contract["stage1_run_authorized"] is False
    assert contract["inference_authorized"] is False
    assert contract["submission_authorized"] is False
    assert contract["create_submission"] is False
    with pytest.raises(RuntimeError, match="inference is disabled"):
        inference.run_inference(config)


def test_compact_candidates_are_self_contained_and_canonical_train_is_adopted():
    train_source = TRAIN_SOURCE.read_text()
    inference_source = INFERENCE_SOURCE.read_text()
    assert "__file__" not in train_source
    assert "__file__" not in inference_source
    assert "from exact_hmm_smoother import" not in train_source
    assert "parent_local_rate_kernel" in train_source
    assert "broad_rate_kernel" in train_source
    assert "_hmm2_symmetric_broad_jump" in train_source
    assert CANONICAL_TRAIN.is_file()
    assert CANONICAL_INFERENCE.is_file()
    assert "broad_rate_kernel" in CANONICAL_TRAIN.read_text()
    assert "broad_rate_kernel" not in CANONICAL_INFERENCE.read_text()
    for heading in (
        "## 1. Imports and immutable contracts",
        "## 3. Fixed32 scope, saved parent, and target-free raw inputs",
        "## 5. Local, broad, and mixture rate-kernel helpers",
        "## 6. Symmetric broad-jump exact forward-backward",
        "## 8. Truth-late Stage 0 readout",
        "## 9. Technical and mechanism gates",
    ):
        assert heading in train_source
    parameters = inspect.signature(
        load_module(
            TRAIN_SOURCE,
            "exp442_train_signature",
        ).freeze_target_free_well
    ).parameters
    assert not {
        "truth",
        "error",
        "episodes",
        "fold",
        "role",
        "cause",
    }.intersection(parameters)


def test_all_stage0_gate_keys_are_consumed_by_one_and_gate(
    train,
    config,
):
    source = inspect.getsource(train.evaluate_stage0_gates)
    technical = config["gates"]["stage0_fixed32"]["technical"]
    mechanism = config["gates"]["stage0_fixed32"]["mechanism"]
    for key in technical:
        assert key in source
    for key in mechanism:
        assert key in source
    assert (
        "all(technical.values())\n        and all(mechanism.values())"
        in source
    )
    assert (
        config["gates"]["stage0_fixed32"]["fail_action"]
        in CONFIG_PATH.read_text()
    )
