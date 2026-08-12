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
    numba_stub.set_num_threads = lambda threads: None
    numba_stub.get_num_threads = lambda: 1
    numba_stub.__version__ = "test-stub"
    sys.modules["numba"] = numba_stub

ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = (
    ROOT
    / "experiments"
    / "exp458_acceleration_state_exact_runtime_engine_audit"
)
TRAIN_SOURCE = (
    EXP_DIR
    / "exp458_acceleration_state_exact_runtime_engine_audit_compact_selfcontained_train.py"
)
INFERENCE_SOURCE = (
    EXP_DIR
    / "exp458_acceleration_state_exact_runtime_engine_audit_compact_selfcontained_inference.py"
)
PARENT_SOURCE = (
    ROOT
    / "experiments"
    / "exp444_acceleration_state_exact_hmm"
    / "exp444_acceleration_state_exact_hmm_compact_selfcontained_train.py"
)
CONFIG_PATH = EXP_DIR / "config.yaml"
CANONICAL_TRAIN = (
    EXP_DIR / "exp458_acceleration_state_exact_runtime_engine_audit_train.ipynb"
)
EXPECTED_SCIENTIFIC_SHA = (
    "f4a0bbbcc8b9cb44a55cff29e07f49ed251e11a896b3e877b4e2d6f9d08f4972"
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
    return load_module(TRAIN_SOURCE, "exp458_train_test")


@pytest.fixture(scope="module")
def inference():
    return load_module(INFERENCE_SOURCE, "exp458_inference_test")


@pytest.fixture(scope="module")
def parent():
    return load_module(PARENT_SOURCE, "exp444_parent_for_exp458_test")


@pytest.fixture(scope="module")
def config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())


def synthetic_prepared(
    fixed: dict,
    *,
    rows: int = 4,
    positions: int = 7,
    rates: int = 5,
) -> dict:
    grid = (
        12_000.0
        + np.arange(positions, dtype=np.float64)
        * float(fixed["position_grid_step_ft"])
    )
    rate_grid = np.linspace(-0.04, 0.04, rates, dtype=np.float64)
    x = np.linspace(-1.0, 1.0, positions)
    emission = np.vstack(
        [
            -0.5 * ((x - 0.15 * np.sin(index)) / 0.44) ** 2
            for index in range(rows)
        ]
    ).astype(np.float32)
    return {
        "emission_ll": emission,
        "dm": np.asarray([1.0, 10.0, 10.0, 12.0], dtype=np.float64)[:rows],
        "dz": np.linspace(-0.2, 0.3, rows, dtype=np.float64),
        "grid": grid,
        "rates": rate_grid,
        "start_p": float(positions // 2),
        "r0": 0.0,
        "eval_index": np.arange(100, 100 + rows, dtype=np.int64),
        "raw_gr_missing": np.zeros(rows, dtype=bool),
    }


def test_stage0a_closed_execution_contract(train, config):
    counts = train.validate_execution_contract(
        config,
        require_run_authorization=False,
    )
    assert counts == {
        "scientific_variants": 1,
        "runtime_engine_candidates": 1,
        "stage0a_repeat_count": 2,
        "stage0a_candidate_hmm_well_runs_per_repeat": 4,
        "stage0a_total_candidate_hmm_well_runs": 8,
        "parent_control_hmm_well_runs": 0,
        "lightgbm_configs": 0,
        "trained_ml_folds": 0,
        "boosters": 0,
        "fitted_models": 0,
        "pf_runs": 0,
        "beam_runs": 0,
        "gpu_runs": 0,
    }
    assert config["design"]["implementation_authorized"] is True
    assert config["execution"]["implementation_authorized"] is True
    assert config["execution"]["canonical_notebook_adoption_authorized"] is True
    assert config["execution"]["kaggle_package_authorized"] is True
    assert config["execution"]["stage0a_run_authorized"] is False
    assert config["execution"]["run_hmm"] is False
    assert config["execution"]["create_prediction"] is False
    assert config["execution"]["create_submission"] is False
    assert (
        config["execution"]["selected_stage"]
        == "stage0a_fixed4_runtime_equivalence_completed_fail_closed"
    )
    with pytest.raises(RuntimeError, match="selected_stage"):
        train.validate_execution_contract(
            config,
            require_run_authorization=True,
        )


def test_scientific_contract_is_exactly_exp444(train, config):
    contract = train.validate_scientific_contract(config)
    observed = hashlib.sha256(
        train.stable_json_bytes(contract)
    ).hexdigest()
    assert observed == EXPECTED_SCIENTIFIC_SHA
    assert observed == config["scientific_contract"]["expected_sha256"]

    broken = deepcopy(config)
    broken["model"]["acceleration_state"]["values"][2] = 0.001
    with pytest.raises(ValueError, match="acceleration contract"):
        train.validate_scientific_contract(broken)


def test_runtime_engine_contract_pins_float64_cache_and_outer4(train, config):
    contract = train.runtime_engine_contract(config)
    assert contract["dtype"] == "float64"
    assert contract["joint_dense_transition_materialized"] is False
    assert contract["delta_md_ou_cache"] == {
        "enabled": True,
        "key": "exact_float64_bit_pattern_within_well",
        "rounding_or_quantization": False,
    }
    assert contract["parallel"]["outer_well_workers"] == 4
    assert contract["parallel"]["numba_threads_per_worker"] == 1
    assert contract["gpu"] is False

    broken = deepcopy(config)
    broken["runtime_engine"]["dtype"] = "float32"
    with pytest.raises(ValueError, match="float64"):
        train.runtime_engine_contract(broken)


def test_exact_bit_delta_md_cache_does_not_quantize(train, config):
    report = train.exact_cache_kernel_contract(
        config["model"]["fixed_from_exp441"],
        config["model"]["acceleration_state"],
    )
    assert report["pass"] is True
    assert report["maximum_abs_error"] <= 1.0e-12
    assert report["nextafter_bit_pattern_is_distinct"] is True
    assert report["duplicate_bit_pattern_is_reused"] is True
    assert report["unique_key_count"] == 3


def test_scaled_engine_matches_dense_reference(train, config):
    report = train.brute_force_posterior_contract(
        config["model"]["fixed_from_exp441"],
        config["model"]["acceleration_state"],
    )
    assert report["pass"] is True
    assert report["posterior_prediction_max_abs_error"] <= 1.0e-6
    assert report["posterior_acceleration_max_abs_error"] <= 1.0e-7


def test_scaled_engine_matches_exp444_log_engine(
    train,
    parent,
    config,
):
    fixed = config["model"]["fixed_from_exp441"]
    acceleration = config["model"]["acceleration_state"]
    prepared = synthetic_prepared(fixed)
    candidate = train.run_acceleration_state_hmm(
        prepared,
        fixed,
        acceleration,
    )
    reference = parent.run_acceleration_state_hmm(
        prepared,
        fixed,
        acceleration,
    )
    np.testing.assert_allclose(
        candidate["posterior_mean"],
        reference["posterior_mean"],
        rtol=0.0,
        atol=1.0e-5,
    )
    np.testing.assert_allclose(
        candidate["posterior_std"],
        reference["posterior_std"],
        rtol=0.0,
        atol=1.0e-5,
    )
    np.testing.assert_allclose(
        candidate["posterior_acceleration"],
        reference["posterior_acceleration"],
        rtol=0.0,
        atol=1.0e-7,
    )
    for key in (
        "predictive_rate_mean",
        "filtered_rate_mean",
        "posterior_rate_mean",
        "posterior_rate_std",
    ):
        np.testing.assert_allclose(
            candidate[key],
            reference[key],
            rtol=0.0,
            atol=5.0e-6,
        )
    assert candidate["prediction_sha256"] == (
        train.run_acceleration_state_hmm(
            prepared,
            fixed,
            acceleration,
        )["prediction_sha256"]
    )


def test_position_kernel_and_worker_thread_guards(train, config):
    position = train.position_kernel_contract(
        config["model"]["fixed_from_exp441"]
    )
    assert position["pass"] is True
    assert position["position_kernel_parent_max_abs_error"] <= 1.0e-12
    environment = train.apply_single_thread_worker_guard()
    assert set(environment) == set(train.THREAD_ENVIRONMENT_KEYS)
    assert set(environment.values()) == {"1"}
    assert train.get_num_threads() == 1


def test_json_serializer_handles_numpy_boolean_gate_values(train):
    payload = {
        "technical": {
            "numpy_true": np.bool_(True),
            "numpy_false": np.bool_(False),
        }
    }
    converted = train.to_jsonable(payload)
    assert converted == {
        "technical": {
            "numpy_true": True,
            "numpy_false": False,
        }
    }
    assert train.json.loads(train.stable_json_bytes(payload)) == converted


def test_gate_consumes_all_frozen_thresholds(train, config):
    source = inspect.getsource(train.evaluate_exp458_stage0a_gates)
    gate = config["gates"]["stage0a_fixed4_runtime_equivalence"]
    for section in ("identity", "numerical", "runtime_reference", "leakage"):
        for key in gate[section]:
            assert key in source
    assert "all(technical.values())" in source


def test_inference_remains_fail_closed_after_stage0a_close(
    inference,
    config,
):
    contract = inference.validate_inference_disabled(config)
    assert contract["implementation_authorized"] is True
    assert contract["canonical_notebook_adoption_authorized"] is True
    assert contract["kaggle_package_authorized"] is True
    assert contract["stage0a_run_authorized"] is False
    assert contract["inference_authorized"] is False
    assert contract["submission_authorized"] is False
    assert config["inference"]["mode"] == "disabled_stage0a_fail_closed"
    with pytest.raises(RuntimeError, match="inference is disabled"):
        inference.run_inference(config)

    source = TRAIN_SOURCE.read_text()
    assert "__file__" not in source
    assert "from settings import" not in source
    assert "_hmm3_acceleration_ou_scaled_probability" in source
    assert CANONICAL_TRAIN.is_file()
    assert (
        "_hmm3_acceleration_ou_scaled_probability"
        in CANONICAL_TRAIN.read_text()
    )
