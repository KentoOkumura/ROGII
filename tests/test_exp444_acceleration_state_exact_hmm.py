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
    numba_stub.set_num_threads = lambda threads: None
    numba_stub.__version__ = "test-stub"
    sys.modules["numba"] = numba_stub

ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "experiments" / "exp444_acceleration_state_exact_hmm"
TRAIN_SOURCE = (
    EXP_DIR
    / "exp444_acceleration_state_exact_hmm_compact_selfcontained_train.py"
)
INFERENCE_SOURCE = (
    EXP_DIR
    / "exp444_acceleration_state_exact_hmm_compact_selfcontained_inference.py"
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
    EXP_DIR / "exp444_acceleration_state_exact_hmm_train.ipynb"
)
CANONICAL_INFERENCE = (
    EXP_DIR / "exp444_acceleration_state_exact_hmm_inference.ipynb"
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
    return load_module(TRAIN_SOURCE, "exp444_train_test")


@pytest.fixture(scope="module")
def inference():
    return load_module(INFERENCE_SOURCE, "exp444_inference_test")


@pytest.fixture(scope="module")
def config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())


def synthetic_inputs(
    rows: int = 6,
) -> tuple[pd.DataFrame, pd.DataFrame]:
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
            "GR": 65.0
            + 8.0 * np.sin((typewell_tvt - 12_000.0) / 18.0),
        }
    )
    return horizontal, typewell


def synthetic_prepared(
    fixed: dict,
    *,
    rows: int = 5,
    positions: int = 9,
    rates: int = 7,
) -> dict:
    grid = (
        12_000.0
        + np.arange(positions, dtype=np.float64)
        * float(fixed["position_grid_step_ft"])
    )
    rate_grid = np.linspace(-0.05, 0.05, rates, dtype=np.float64)
    x = np.linspace(-1.0, 1.0, positions)
    emission = np.vstack(
        [
            -0.5
            * (
                (
                    x - 0.2 * np.sin(index / 2.0)
                )
                / 0.42
            )
            ** 2
            for index in range(rows)
        ]
    ).astype(np.float32)
    return {
        "emission_ll": emission,
        "dm": np.linspace(1.0, 15.0, rows, dtype=np.float64),
        "dz": np.linspace(-0.2, 0.3, rows, dtype=np.float64),
        "grid": grid,
        "rates": rate_grid,
        "start_p": float(positions // 2),
        "r0": 0.0,
        "eval_index": np.arange(100, 100 + rows, dtype=np.int64),
        "raw_gr_missing": np.zeros(rows, dtype=bool),
    }


def test_stage0a_result_is_independent_and_all_runs_fail_closed(
    train,
    config,
):
    observed = train.validate_execution_contract(
        config,
        require_run_authorization=False,
    )
    assert observed == {
        "scientific_variants": 1,
        "stage0a_candidate_hmm_well_runs": 4,
        "stage0b_total_candidate_hmm_well_runs": 32,
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
    assert config["design"]["independent_execution_hypothesis"] is True
    assert config["design"]["implementation_authorized"] is True
    assert config["execution"]["implementation_authorized"] is True
    assert config["execution"]["canonical_notebook_adoption_authorized"] is True
    assert config["execution"]["kaggle_package_authorized"] is True
    assert config["execution"]["stage0a_run_authorized"] is False
    assert config["execution"]["run_hmm"] is False
    assert config["execution"]["create_prediction"] is False
    assert config["execution"]["create_submission"] is False
    assert config["execution"]["stage0b_run_authorized"] is False
    assert config["execution"]["stage1_run_authorized"] is False
    assert config["execution"]["inference_authorized"] is False
    assert config["execution"]["submission_authorized"] is False
    with pytest.raises(RuntimeError, match="Stage 0A"):
        train.validate_execution_contract(
            config,
            require_run_authorization=True,
        )

    broken = deepcopy(config)
    broken["execution"]["stage0b_run_authorized"] = True
    with pytest.raises(ValueError, match="Stage 0B"):
        train.validate_execution_contract(
            broken,
            require_run_authorization=False,
        )


def test_scientific_contract_pins_three_state_acceleration(train, config):
    contract = train.validate_scientific_contract(config)
    fixed = contract["fixed_from_exp441"]
    acceleration = contract["acceleration_state"]
    assert fixed["rate_transition_family"] == (
        "exact_ornstein_uhlenbeck_full_support_bin_integral"
    )
    assert fixed["rate_momentum"] == 0.998
    assert acceleration["values"] == [-0.0005, 0.0, 0.0005]
    assert acceleration["initial_probability"] == [0.0, 1.0, 0.0]
    assert acceleration["transition"]["interior_to_lower"] == 0.08
    assert acceleration["transition"]["interior_stay"] == 0.84
    assert acceleration["transition"]["interior_to_upper"] == 0.08
    assert contract["active_scientific_variants"] == [
        "three_state_persistent_acceleration"
    ]

    broken = deepcopy(config)
    broken["model"]["acceleration_state"]["values"][2] = 0.001
    with pytest.raises(ValueError, match="acceleration contract"):
        train.validate_scientific_contract(broken)


def test_acceleration_transition_boundary_and_initial_prior(train, config):
    acceleration = config["model"]["acceleration_state"]
    matrix = train.acceleration_transition_matrix(acceleration)
    np.testing.assert_allclose(
        matrix,
        np.asarray(
            [
                [0.92, 0.08, 0.0],
                [0.08, 0.84, 0.08],
                [0.0, 0.08, 0.92],
            ]
        ),
        rtol=0.0,
        atol=1.0e-15,
    )
    np.testing.assert_allclose(
        matrix.sum(axis=1),
        np.ones(3),
        rtol=0.0,
        atol=1.0e-15,
    )
    contract = train.acceleration_transition_contract(acceleration)
    assert contract["pass"] is True
    assert contract["acceleration_row_sum_max_error"] <= 1.0e-12


def test_destination_acceleration_orders_ou_rate_means(train, config):
    fixed = config["model"]["fixed_from_exp441"]
    acceleration = config["model"]["acceleration_state"]
    rates = np.linspace(-0.10, 0.10, 81, dtype=np.float64)
    logs = train.precompute_acceleration_ou_log_kernels(
        np.asarray([10.0], dtype=np.float64),
        rates,
        np.asarray(acceleration["values"], dtype=np.float64),
        fixed["sig_r"],
        fixed["rate_momentum"],
    )
    source = len(rates) // 2
    means = []
    for acceleration_index in range(3):
        probabilities = np.exp(logs[0, acceleration_index, source])
        means.append(float(probabilities @ rates / probabilities.sum()))
    assert means[0] < means[1] < means[2]
    assert means[0] == pytest.approx(-0.005, abs=2.0e-4)
    assert means[1] == pytest.approx(0.0, abs=2.0e-4)
    assert means[2] == pytest.approx(0.005, abs=2.0e-4)


def test_zero_acceleration_rate_kernel_is_exp441_parity(train, config):
    contract = train.zero_acceleration_kernel_parity_contract(
        config["model"]["fixed_from_exp441"]
    )
    assert contract["pass"] is True
    assert contract[
        "zero_acceleration_rate_kernel_parity_vs_exp441_max_abs_error"
    ] <= 1.0e-12


def test_prepare_hmm_inputs_preserves_parent_emission(train, config):
    horizontal, typewell = synthetic_inputs()
    fixed = config["model"]["fixed_from_exp441"]
    prepared = train.prepare_hmm_inputs(horizontal, typewell, fixed)
    known = horizontal.loc[horizontal["TVT_input"].notna()]
    typewell_at_known = np.interp(
        known["TVT_input"].to_numpy(np.float64),
        typewell["TVT"].to_numpy(np.float64),
        typewell["GR"].to_numpy(np.float64),
    )
    sigma = float(
        np.clip(
            np.nanstd(
                known["GR"].fillna(0).to_numpy(np.float64)
                - typewell_at_known
            ),
            10.0,
            60.0,
        )
    )
    gr_grid = np.interp(
        prepared["grid"],
        typewell["TVT"].to_numpy(np.float64),
        typewell["GR"].to_numpy(np.float64),
    )
    suffix_gr = (
        horizontal["GR"]
        .interpolate(limit_direction="both")
        .to_numpy(np.float64)[prepared["eval_index"]]
    )
    expected = -0.5 * np.minimum(
        ((suffix_gr[:, None] - gr_grid[None, :]) / sigma) ** 2,
        600.0,
    )
    np.testing.assert_allclose(
        prepared["emission_ll"],
        expected.astype(np.float32),
        rtol=0.0,
        atol=0.0,
    )
    assert prepared["rates"].shape == (41,)


def test_small_hmm_matches_independent_dense_reference(train, config):
    contract = train.brute_force_posterior_contract(
        config["model"]["fixed_from_exp441"],
        config["model"]["acceleration_state"],
    )
    assert contract["pass"] is True
    assert contract["posterior_prediction_max_abs_error"] <= 1.0e-6
    assert contract["posterior_acceleration_max_abs_error"] <= 1.0e-6


def test_target_free_hmm_returns_all_acceleration_diagnostics(train, config):
    fixed = config["model"]["fixed_from_exp441"]
    acceleration = config["model"]["acceleration_state"]
    prepared = synthetic_prepared(fixed)
    result = train.run_acceleration_state_hmm(
        prepared,
        fixed,
        acceleration,
    )
    rows = len(prepared["dm"])
    for key in (
        "posterior_mean",
        "posterior_std",
        "posterior_rate_mean",
        "posterior_rate_std",
        "posterior_acceleration_mean",
        "posterior_acceleration_nonzero_mass",
        "predictive_rate_mean",
        "filtered_rate_mean",
        "predictive_acceleration_mean",
        "filtered_acceleration_mean",
    ):
        values = np.asarray(result[key])
        assert values.shape == (rows,)
        assert np.isfinite(values).all()
    posterior = np.asarray(result["posterior_acceleration"])
    assert posterior.shape == (rows, 3)
    np.testing.assert_allclose(
        posterior.sum(axis=1),
        np.ones(rows),
        rtol=0.0,
        atol=1.0e-9,
    )
    assert np.all(result["posterior_acceleration_nonzero_mass"] > 0.0)
    assert result["maximum_normalization_error"] <= 1.0e-6
    for key in (
        "joint_transition_sha256",
        "prediction_sha256",
        "acceleration_posterior_sha256",
        "diagnostic_sha256",
    ):
        assert len(result[key]) == 64


def test_fixed4_selector_is_identity_only_sha_order(train, config):
    ledger = train.LeakageLedger(expected_wells=4)
    selected, report = train.select_stage0a_wells(config, ledger)
    manifest = pd.read_csv(MANIFEST_PATH, usecols=["well"], dtype={"well": str})
    expected = sorted(
        manifest["well"].tolist(),
        key=lambda well: (train.stage0a_identity_hash(well), well),
    )[:4]
    assert selected == expected
    assert report["columns_read"] == ["well"]
    assert ledger.identity_rows_read == 32
    assert ledger.forbidden_reads_before_all_freeze == 0
    assert len(set(selected)) == 4

    observed_sha = hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest()
    assert observed_sha == config["data"]["fixed32_manifest"]["expected_sha256"]


def test_leakage_guard_blocks_forbidden_read_until_fixed4_freeze(train):
    ledger = train.LeakageLedger(expected_wells=2)
    ledger.freeze(
        "a",
        joint_transition_sha256="j1",
        prediction_sha256="p1",
        acceleration_posterior_sha256="a1",
        diagnostic_sha256="d1",
    )
    with pytest.raises(RuntimeError, match="truth"):
        ledger.record_forbidden("truth", 3)
    assert ledger.forbidden_reads_before_all_freeze == 3
    ledger.freeze(
        "b",
        joint_transition_sha256="j2",
        prediction_sha256="p2",
        acceleration_posterior_sha256="a2",
        diagnostic_sha256="d2",
    )
    assert ledger.all_frozen


def test_deterministic_gzip_round_trip_sha(train, tmp_path):
    frame = pd.DataFrame(
        {
            "well": ["a", "a"],
            "row_idx": [1, 2],
            "acceleration_zero_mass": [0.8, 0.7],
            "posterior_acceleration_mean": [0.0, 0.0001],
        }
    )
    artifact = train.write_deterministic_gzip_csv(
        tmp_path / "posterior.csv.gz",
        frame,
    )
    assert artifact["logical_sha256"] == artifact[
        "readback_logical_sha256"
    ]
    assert artifact["decompressed_sha256"]


def test_inference_remains_fail_closed(inference, config):
    contract = inference.validate_inference_disabled(config)
    assert contract["implementation_authorized"] is True
    assert contract["canonical_notebook_adoption_authorized"] is True
    assert contract["kaggle_package_authorized"] is True
    assert contract["stage0a_run_authorized"] is False
    assert contract["stage0b_run_authorized"] is False
    assert contract["stage1_run_authorized"] is False
    assert contract["inference_authorized"] is False
    assert contract["submission_authorized"] is False
    with pytest.raises(RuntimeError, match="inference is disabled"):
        inference.run_inference(config)


def test_compact_candidates_are_self_contained_and_canonical_train_is_adopted():
    train_source = TRAIN_SOURCE.read_text()
    inference_source = INFERENCE_SOURCE.read_text()
    assert "__file__" not in train_source
    assert "__file__" not in inference_source
    assert "from settings import" not in train_source
    assert "from exact_hmm_smoother import" not in train_source
    assert "_hmm3_acceleration_ou" in train_source
    assert "precompute_acceleration_ou_log_kernels" in train_source
    assert CANONICAL_TRAIN.is_file()
    assert CANONICAL_INFERENCE.is_file()
    assert "_hmm3_acceleration_ou" in CANONICAL_TRAIN.read_text()
    assert "_hmm3_acceleration_ou" not in CANONICAL_INFERENCE.read_text()


def test_all_stage0a_gate_keys_are_consumed_by_one_and_gate(train, config):
    source = inspect.getsource(train.evaluate_stage0a_gates)
    technical = config["gates"]["stage0a_fixed4_runtime"]["technical"]
    for key in technical:
        assert key in source
    assert "all(technical.values())" in source
