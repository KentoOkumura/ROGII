from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "experiments" / "exp446_persistent_tvt_rate_exact_hmm"
TRAIN_SOURCE = (
    EXP_DIR
    / "exp446_persistent_tvt_rate_exact_hmm_compact_selfcontained_train.py"
)
INFERENCE_SOURCE = (
    EXP_DIR
    / "exp446_persistent_tvt_rate_exact_hmm_compact_selfcontained_inference.py"
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
    EXP_DIR / "exp446_persistent_tvt_rate_exact_hmm_train.ipynb"
)
CANONICAL_INFERENCE = (
    EXP_DIR / "exp446_persistent_tvt_rate_exact_hmm_inference.ipynb"
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
    return load_module(TRAIN_SOURCE, "exp446_train_test")


@pytest.fixture(scope="module")
def inference():
    return load_module(INFERENCE_SOURCE, "exp446_inference_test")


@pytest.fixture(scope="module")
def config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())


def test_completed_stage0_contract_is_fail_closed_and_bounded(train, config):
    observed = train.validate_execution_contract(
        config,
        require_run_authorization=False,
    )
    assert observed == {
        "active_scientific_variants": 1,
        "stage0_treatment_hmm_well_runs": 32,
        "stage1_max_treatment_hmm_well_runs": 773,
        "parent_control_hmm_reruns_stage0": 0,
        "parent_control_hmm_reruns_stage1": 0,
        "lightgbm_configs": 0,
        "trained_ml_folds": 0,
        "boosters": 0,
        "fitted_models": 0,
        "pf_runs": 0,
        "beam_runs": 0,
        "gpu_runs": 0,
    }
    assert config["experiment"]["status"] == "stage0_fail_closed"
    assert config["design"]["implementation_authorized"] is True
    assert config["design"]["canonical_notebook_adoption_authorized"] is True
    assert config["design"]["kaggle_package_authorized"] is True
    assert config["design"]["stage0_run_authorized"] is True
    assert config["design"]["stage0_rerun_authorized"] is False
    assert config["design"]["stage1_run_authorized"] is False
    assert config["execution"]["selected_stage"] == (
        "completed_stage0_fail_closed"
    )
    assert config["execution"]["run_hmm"] is False
    assert config["execution"]["create_prediction"] is False
    assert config["execution"]["create_submission"] is False
    with pytest.raises(RuntimeError, match="selected_stage"):
        train.validate_execution_contract(
            config,
            require_run_authorization=True,
        )

    unlocked = deepcopy(config)
    unlocked["execution"]["selected_stage"] = "stage0_fixed32"
    with pytest.raises(RuntimeError, match="run_hmm remains fail-closed"):
        train.validate_execution_contract(
            unlocked,
            require_run_authorization=True,
        )

    unlocked = deepcopy(config)
    unlocked["design"]["stage1_run_authorized"] = True
    with pytest.raises(ValueError, match="Stage 1"):
        train.validate_execution_contract(
            unlocked,
            require_run_authorization=False,
        )


def test_scientific_contract_pins_tvt_rate_single_change(train, config):
    contract = train.validate_scientific_contract(config)
    candidate = contract["candidate_state"]
    fixed = contract["fixed_from_exp209"]
    assert candidate["rate_definition"] == "q=dTVT/dMD"
    assert candidate["prefix_tail_steps"] == 50
    assert candidate["position_mean_formula"] == "q_destination*delta_MD"
    assert candidate["position_includes_delta_z_term"] is False
    assert fixed["n_rates"] == 41
    assert fixed["rate_span_min"] == 0.10
    assert fixed["rate_initial_margin"] == 0.04
    assert fixed["momentum"] == 0.998
    assert fixed["sig_r"] == 0.002
    assert contract["active_scientific_variants"] == [
        "persistent_tvt_rate"
    ]

    broken = deepcopy(config)
    broken["model"]["candidate_state"][
        "position_includes_delta_z_term"
    ] = True
    with pytest.raises(ValueError, match="candidate contract"):
        train.validate_scientific_contract(broken)


def test_prefix_initialization_uses_tvt_rate_not_u_rate(train):
    md = np.arange(61, dtype=np.float64) * 10.0
    tvt = 12_000.0 + 0.025 * md
    z = 8_000.0 + 0.01 * md
    prefix = pd.DataFrame({"MD": md, "Z": z, "TVT_input": tvt})
    q0, rows, q_steps = train.robust_initial_rate(
        prefix,
        window_rows=50,
        include_delta_z=False,
    )
    u0, _, u_steps = train.robust_initial_rate(
        prefix,
        window_rows=50,
        include_delta_z=True,
    )
    assert rows == 50
    assert q_steps == u_steps == 49
    assert q0 == pytest.approx(0.025, abs=1.0e-12)
    assert u0 == pytest.approx(0.035, abs=1.0e-12)


def test_parent_local_rate_kernel_matches_exp209_formula(train, config):
    rates = np.linspace(-0.10, 0.10, 41, dtype=np.float64)
    delta_md = 17.5
    sig_r = config["model"]["fixed_from_exp209"]["sig_r"]
    momentum = config["model"]["fixed_from_exp209"]["momentum"]
    observed = train.rate_kernel_probabilities(
        rates,
        delta_md,
        sig_r,
        momentum,
    )
    rate_step = rates[1] - rates[0]
    variance_cells = (sig_r * np.sqrt(delta_md) / rate_step) ** 2
    expected = np.empty_like(observed)
    for index, rate in enumerate(rates):
        mean_move = -(1.0 - momentum) * rate * delta_md / rate_step
        plus = max(0.5 * (variance_cells + mean_move), 1.0e-12)
        minus = max(0.5 * (variance_cells - mean_move), 1.0e-12)
        if plus + minus > 0.9:
            scale = 0.9 / (plus + minus)
            plus *= scale
            minus *= scale
        expected[index] = [minus, 1.0 - plus - minus, plus]
    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        observed.sum(axis=1),
        1.0,
        rtol=0.0,
        atol=1.0e-15,
    )


def test_constant_z_parent_candidate_contract_is_numerically_equal(
    train,
    config,
):
    contract = train.constant_z_parent_parity_contract(
        config["model"]["fixed_from_exp209"]
    )
    assert contract["kernel_max_abs_error"] <= 1.0e-12
    assert contract["posterior_prediction_max_abs_error"] <= 1.0e-6
    assert contract["initial_rate_abs_error"] <= 1.0e-12
    assert contract["rate_grid_max_abs_error"] <= 1.0e-12
    assert contract["position_posterior_max_abs_error"] <= 1.0e-8
    assert contract["rate_posterior_max_abs_error"] <= 1.0e-8


def test_candidate_transition_is_invariant_to_delta_z(train, config):
    fixed = config["model"]["fixed_from_exp209"]
    positions = 9
    rows = 4
    base = {
        "emission_ll": np.zeros((rows, positions), dtype=np.float32),
        "dm": np.asarray([1.0, 3.0, 7.0, 12.0], dtype=np.float64),
        "grid": 12_000.0 + np.arange(positions) * 0.35,
        "rates": np.linspace(-0.04, 0.04, 7),
        "start_p": 4.0,
        "r0": 0.01,
    }
    constant_z = {
        **base,
        "dz": np.zeros(rows, dtype=np.float64),
    }
    variable_z = {
        **base,
        "dz": np.asarray([0.0, 0.7, -1.1, 0.2], dtype=np.float64),
    }
    first = train.run_persistent_tvt_rate_hmm(constant_z, fixed)
    second = train.run_persistent_tvt_rate_hmm(variable_z, fixed)
    np.testing.assert_array_equal(
        first["posterior_position"],
        second["posterior_position"],
    )
    np.testing.assert_array_equal(
        first["posterior_rate"],
        second["posterior_rate"],
    )
    assert first["transition_kernel_sha256"] == (
        second["transition_kernel_sha256"]
    )


def test_small_state_dense_reference_and_sha_contracts(train, config):
    fixed = config["model"]["fixed_from_exp209"]
    brute = train.brute_force_posterior_contract(fixed)
    assert brute["pass"] is True
    assert brute["maximum_abs_error"] <= 1.0e-6

    prepared = {
        "emission_ll": np.zeros((3, 7), dtype=np.float32),
        "dm": np.asarray([1.0, 2.0, 4.0], dtype=np.float64),
        "dz": np.asarray([0.2, -0.3, 0.1], dtype=np.float64),
        "grid": 12_000.0 + np.arange(7) * 0.35,
        "rates": np.linspace(-0.03, 0.03, 5),
        "start_p": 3.0,
        "r0": 0.0,
    }
    result = train.run_persistent_tvt_rate_hmm(prepared, fixed)
    assert result["maximum_normalization_error"] <= 1.0e-6
    for key in (
        "rate_grid_sha256",
        "rate_kernel_sha256",
        "transition_kernel_sha256",
        "posterior_sha256",
        "prediction_sha256",
        "diagnostic_sha256",
    ):
        assert len(result[key]) == 64


def test_position_kernel_and_truth_late_ledger_contracts(train, config):
    contract = train.position_kernel_contract(
        config["model"]["fixed_from_exp209"]
    )
    assert contract["pass"] is True
    assert contract["position_kernel_max_abs_error"] <= 1.0e-12
    assert contract["position_kernel_row_sum_max_error"] <= 1.0e-12
    assert contract["position_edge_residual_max_abs_ft"] <= 1.0e-12

    ledger = train.LeakageLedger(expected_wells=1)
    with pytest.raises(RuntimeError, match="truth"):
        ledger.record_truth_late(1)
    assert ledger.forbidden_reads_before_all_freeze == 1
    ledger.freeze(
        "well",
        rate_grid_sha256="g",
        rate_kernel_sha256="r",
        transition_kernel_sha256="t",
        posterior_sha256="o",
        prediction_sha256="p",
        diagnostic_sha256="d",
    )
    ledger.record_truth_late(2)
    ledger.record_role_fold_late(1)
    assert ledger.all_frozen
    assert ledger.truth_rows_after_all_freeze == 2


def test_fixed32_manifest_is_sha_pinned_target_free_and_balanced(
    train,
    config,
):
    observed = hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest()
    spec = config["data"]["stage0_manifest"]
    assert observed == spec["expected_sha256"]
    manifest = pd.read_csv(MANIFEST_PATH, dtype={"well": str})
    assert len(manifest) == 32
    assert manifest["well"].nunique() == 32
    assert int(manifest["suffix_rows"].sum()) == 156_088
    assert manifest["role"].value_counts().to_dict() == {
        "persistent": 16,
        "control": 16,
    }
    ledger = train.LeakageLedger(expected_wells=32)
    wells, report = train.load_fixed32_scope(config, ledger)
    assert len(wells) == len(set(wells)) == 32
    assert report["sha256"] == observed
    assert ledger.scope_rows == 32
    assert ledger.forbidden_reads_before_all_freeze == 0


def test_inference_remains_fail_closed(inference, config):
    contract = inference.validate_inference_disabled(config)
    assert contract["implementation_authorized"] is True
    assert contract["canonical_notebook_adoption_authorized"] is True
    assert contract["kaggle_package_authorized"] is True
    assert contract["stage0_run_authorized"] is True
    assert contract["inference_authorized"] is False
    assert contract["submission_authorized"] is False
    with pytest.raises(RuntimeError, match="inference is disabled"):
        inference.run_inference(config)


def test_compact_candidate_is_self_contained_and_canonical_is_adopted():
    source = TRAIN_SOURCE.read_text()
    assert "__file__" not in source
    assert "from settings import" not in source
    assert "from exact_hmm_smoother import" not in source
    assert "def robust_initial_rate(" in source
    assert "def rate_kernel_probabilities(" in source
    assert "def _hmm2_persistent_rate(" in source
    assert "def constant_z_parent_parity_contract(" in source
    assert "def run_stage0(" in source
    headings = [
        line for line in source.splitlines() if line.startswith("# ## ")
    ]
    assert len(headings) == 11
    assert len(source.splitlines()) >= 2_500
    assert CANONICAL_TRAIN.is_file()
    canonical = json.loads(CANONICAL_TRAIN.read_text())
    canonical_text = "\n".join(
        "".join(cell.get("source", []))
        for cell in canonical["cells"]
    )
    assert "Metrics scaffold" not in canonical_text
    assert "def _hmm2_persistent_rate(" in canonical_text
    assert "def constant_z_parent_parity_contract(" in canonical_text
    assert "def run_stage0(" in canonical_text
    assert len(canonical["cells"]) >= 20

    inference_notebook = json.loads(CANONICAL_INFERENCE.read_text())
    inference_text = "\n".join(
        "".join(cell.get("source", []))
        for cell in inference_notebook["cells"]
    )
    assert "def validate_inference_disabled(" in inference_text
    assert "def run_inference(" in inference_text


def test_every_preregistered_gate_key_is_consumed(train, config):
    source = inspect.getsource(train.evaluate_stage0_gates)
    technical = config["validation"]["stage0"]["technical"]
    mechanism = config["validation"]["stage0"]["mechanism"]
    for key in (*technical, *mechanism):
        assert key in source
    assert "all(technical.values())" in source
    assert "all(mechanism.values())" in source
