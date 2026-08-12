from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.util
import inspect
import sys
import types
from copy import deepcopy
from pathlib import Path
from statistics import NormalDist

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

ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = (
    ROOT
    / "experiments"
    / "exp441_full_support_ou_rate_transition_hmm"
)
TRAIN_SOURCE = (
    EXP_DIR
    / "exp441_full_support_ou_rate_transition_hmm_"
    "compact_selfcontained_train.py"
)
INFERENCE_SOURCE = (
    EXP_DIR
    / "exp441_full_support_ou_rate_transition_hmm_"
    "compact_selfcontained_inference.py"
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
    EXP_DIR / "exp441_full_support_ou_rate_transition_hmm_train.ipynb"
)
CANONICAL_INFERENCE = (
    EXP_DIR / "exp441_full_support_ou_rate_transition_hmm_inference.ipynb"
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
    return load_module(TRAIN_SOURCE, "exp441_train_test")


@pytest.fixture(scope="module")
def inference():
    return load_module(INFERENCE_SOURCE, "exp441_inference_test")


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


def test_stage0_execution_contract_unlocks_only_fixed32(
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
    assert config["experiment"]["status"] in {
        "stage0_authorized_pending_run",
        "kaggle_v1_running_stage0",
        "stage0_all_gates_pass_pending_separate_stage1_approval",
        "stage0_fail_closed",
        "stage0_error",
    }
    assert config["execution"]["implementation_authorized"] is True
    assert (
        config["execution"]["canonical_notebook_adoption_authorized"]
        is True
    )
    assert config["execution"]["kaggle_package_authorized"] is True
    assert config["execution"]["stage0_run_authorized"] is True
    assert config["execution"]["stage1_run_authorized"] is False
    assert config["execution"]["inference_authorized"] is False
    assert config["execution"]["submission_authorized"] is False
    assert (
        train.validate_execution_contract(
            config,
            require_run_authorization=True,
        )
        == observed
    )

    disabled = deepcopy(config)
    disabled["execution"]["stage0_run_authorized"] = False
    with pytest.raises(
        RuntimeError,
        match="does not authorize Stage 0 execution",
    ):
        train.validate_execution_contract(
            disabled,
            require_run_authorization=True,
        )

    broken = deepcopy(config)
    broken["execution"]["stage1_run_authorized"] = True
    with pytest.raises(ValueError, match="Stage 1"):
        train.validate_execution_contract(
            broken,
            require_run_authorization=False,
        )


def test_scientific_contract_pins_exact_ou_single_change(train, config):
    contract = train.validate_scientific_contract(config)
    assert contract["fixed_from_exp209"]["momentum"] == 0.998
    assert contract["fixed_from_exp209"]["sig_r"] == 0.002
    assert contract["fixed_from_exp209"]["position_mean_formula"] == (
        "r_destination*delta_MD-delta_Z"
    )
    candidate = contract["candidate_rate_transition"]
    assert candidate["family"] == "exact_ornstein_uhlenbeck"
    assert candidate["numerical_dtype"] == "float64"
    assert candidate["outer_tail_policy"].endswith(
        "without_renormalization"
    )
    assert contract["active_scientific_variants"] == [
        "full_support_exact_ou"
    ]

    broken = deepcopy(config)
    broken["model"]["candidate_rate_transition"][
        "outer_tail_policy"
    ] = "renormalize"
    with pytest.raises(ValueError, match="candidate contract"):
        train.validate_scientific_contract(broken)


def test_prepare_hmm_inputs_preserves_exp209_emission(train, config):
    horizontal, typewell = synthetic_inputs()
    fixed = config["model"]["fixed_from_exp209"]
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
        .fillna(typewell["GR"].mean())
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


def test_exact_ou_parameters_and_zero_kappa_limit(train):
    kappa, decay, variance = train.ou_conditional_parameters(
        17.5,
        0.002,
        0.998,
    )
    expected_kappa = -np.log(0.998)
    assert kappa == pytest.approx(expected_kappa, abs=1.0e-15)
    assert decay == pytest.approx(
        np.exp(-expected_kappa * 17.5),
        abs=1.0e-15,
    )
    assert variance == pytest.approx(
        0.002**2
        * (1.0 - np.exp(-2.0 * expected_kappa * 17.5))
        / (2.0 * expected_kappa),
        abs=1.0e-18,
    )
    _, unit_decay, brownian_variance = (
        train.ou_conditional_parameters(17.5, 0.002, 1.0)
    )
    assert unit_decay == 1.0
    assert brownian_variance == pytest.approx(
        0.002**2 * 17.5,
        abs=1.0e-18,
    )


def test_full_support_kernel_matches_independent_normal_cdf_and_drops_tail(
    train,
    config,
):
    rates = np.linspace(-0.10, 0.10, 41)
    delta_md = 25.0
    fixed = config["model"]["fixed_from_exp209"]
    kernel = train.full_support_ou_rate_kernel(
        rates,
        delta_md,
        fixed["sig_r"],
        fixed["momentum"],
    )
    edges = np.r_[
        rates[0] - 0.5 * (rates[1] - rates[0]),
        0.5 * (rates[:-1] + rates[1:]),
        rates[-1] + 0.5 * (rates[-1] - rates[-2]),
    ]
    kappa = -np.log(fixed["momentum"])
    decay = np.exp(-kappa * delta_md)
    variance = (
        fixed["sig_r"] ** 2
        * (1.0 - np.exp(-2.0 * kappa * delta_md))
        / (2.0 * kappa)
    )
    source = 40
    normal = NormalDist(mu=decay * rates[source], sigma=np.sqrt(variance))
    expected = np.asarray(
        [
            normal.cdf(edges[index + 1])
            - normal.cdf(edges[index])
            for index in range(len(rates))
        ]
    )
    np.testing.assert_allclose(
        kernel[source],
        expected,
        rtol=0.0,
        atol=2.0e-16,
    )
    assert kernel[source].sum() < 1.0
    assert kernel[source, source - 2] > 0.0
    assert kernel.shape == (41, 41)


def test_ou_numeric_audit_and_position_kernel_parity(train, config):
    fixed = config["model"]["fixed_from_exp209"]
    rates = np.linspace(-0.10, 0.10, 41)
    delta_md = np.asarray([1.0, 7.5, 25.0], dtype=np.float64)
    logs = train.precompute_full_support_ou_log_kernels(
        delta_md,
        rates,
        fixed["sig_r"],
        fixed["momentum"],
    )
    audit = train.ou_kernel_numeric_audit(
        logs,
        delta_md,
        rates,
        fixed["sig_r"],
        fixed["momentum"],
    )
    assert audit["analytic_in_support_mass_max_abs_error"] <= 1.0e-12
    assert audit["interior_conditional_mean_max_abs_error"] <= 1.0e-12
    assert (
        audit["interior_conditional_variance_max_abs_error"]
        <= 1.0e-12
    )
    assert audit["minimum_transition_row_mass"] < 1.0
    parity = train.position_kernel_parity_contract(fixed)
    assert parity["pass"] is True
    assert parity["maximum_absolute_error"] <= 1.0e-12


def test_small_hmm_matches_dense_brute_force_reference(train, config):
    contract = train.brute_force_posterior_contract(
        config["model"]["fixed_from_exp209"]
    )
    assert contract["pass"] is True
    assert contract["posterior_prediction_max_abs_error"] <= 1.0e-6


def test_target_free_hmm_returns_rate_diagnostics_and_actual_kernel_sha(
    train,
    config,
):
    fixed = config["model"]["fixed_from_exp209"]
    rows = 5
    positions = 9
    prepared = {
        "emission_ll": np.vstack(
            [
                -0.5
                * (
                    (
                        np.linspace(-1.0, 1.0, positions)
                        - 0.2 * np.sin(index)
                    )
                    / 0.4
                )
                ** 2
                for index in range(rows)
            ]
        ).astype(np.float32),
        "dm": np.asarray([1.0, 3.0, 7.0, 12.0, 20.0]),
        "dz": np.asarray([0.0, 0.1, -0.1, 0.2, 0.0]),
        "grid": 12_000.0 + np.arange(positions) * 0.35,
        "rates": np.linspace(-0.05, 0.05, 7),
        "start_p": 4.0,
        "r0": 0.01,
    }
    result = train.run_full_support_ou_hmm(prepared, fixed)
    for key in (
        "posterior_mean",
        "posterior_std",
        "predictive_rate_mean",
        "filtered_rate_mean",
        "posterior_rate_mean",
        "posterior_rate_std",
        "posterior_rate_edge_mass",
    ):
        values = np.asarray(result[key])
        assert values.shape == (rows,)
        assert np.isfinite(values).all()
    assert len(result["transition_kernel_sha256"]) == 64
    assert result["maximum_normalization_error"] <= 1.0e-6


def test_leakage_ledger_blocks_truth_identity_episode_and_cause_until_freeze(
    train,
):
    ledger = train.LeakageLedger(expected_wells=2)
    ledger.freeze(
        "a",
        transition_kernel_sha256="k1",
        prediction_sha256="p1",
        diagnostic_sha256="d1",
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
        transition_kernel_sha256="k2",
        prediction_sha256="p2",
        diagnostic_sha256="d2",
    )
    ledger.record_role_fold_late(2)
    ledger.record_truth_late(5)
    ledger.record_episode_late(1)
    ledger.record_cause_late(1)
    assert ledger.all_frozen


def test_zero_directed_under_response_definition(train):
    true_rate = np.asarray([0.04, -0.04, 0.04, -0.04, 0.0])
    decoded = np.asarray([0.02, -0.01, -0.02, -0.06, 0.0])
    np.testing.assert_array_equal(
        train.zero_directed_under_response_mask(true_rate, decoded),
        np.asarray([True, True, False, False, False]),
    )


def test_fixed32_manifest_is_sha_pinned_and_late_identity_is_balanced(
    config,
):
    observed = hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest()
    assert observed == config["data"]["fixed32_manifest"]["expected_sha256"]
    manifest = pd.read_csv(MANIFEST_PATH, dtype={"well": str})
    assert len(manifest) == 32
    assert manifest["well"].nunique() == 32
    assert int(manifest["suffix_rows"].sum()) == 156_088
    assert manifest["role"].value_counts().to_dict() == {
        "persistent": 16,
        "control": 16,
    }
    assert manifest.groupby("fold").size().to_dict() == {
        0: 8,
        1: 6,
        2: 6,
        3: 6,
        4: 6,
    }


def test_deterministic_gzip_round_trip_sha(train, tmp_path):
    frame = pd.DataFrame(
        {
            "well": ["a", "a"],
            "row_idx": [1, 2],
            "filtered_rate_mean": [0.0125, -0.0075],
            "posterior_rate_edge_mass": [0.001, 0.002],
        }
    )
    artifact = train.write_deterministic_gzip_csv(
        tmp_path / "diagnostic.csv.gz",
        frame,
    )
    assert artifact["logical_sha256"] == artifact[
        "readback_logical_sha256"
    ]
    assert artifact["logical_sha256"] == artifact[
        "decompressed_sha256"
    ]


def test_inference_remains_fail_closed(inference, config):
    contract = inference.validate_inference_disabled(config)
    assert contract["implementation_authorized"] is True
    assert contract["inference_authorized"] is False
    assert contract["submission_authorized"] is False
    with pytest.raises(RuntimeError, match="inference is disabled"):
        inference.run_inference(config)


def test_compact_candidates_are_self_contained_and_canonical_train_is_adopted():
    train_source = TRAIN_SOURCE.read_text()
    inference_source = INFERENCE_SOURCE.read_text()
    assert "__file__" not in train_source
    assert "__file__" not in inference_source
    assert "from exact_hmm_smoother import" not in train_source
    assert "precompute_full_support_ou_log_kernels" in train_source
    assert "_hmm2_full_support_ou" in train_source
    assert CANONICAL_TRAIN.is_file()
    assert CANONICAL_INFERENCE.is_file()
    assert "precompute_full_support_ou_log_kernels" in (
        CANONICAL_TRAIN.read_text()
    )
    assert "_hmm2_full_support_ou" in CANONICAL_TRAIN.read_text()


def test_all_stage0_gate_keys_are_consumed_by_one_and_gate(train, config):
    source = inspect.getsource(train.evaluate_stage0_gates)
    technical = config["gates"]["stage0_fixed32"]["technical"]
    mechanism = config["gates"]["stage0_fixed32"]["mechanism"]
    for key in (*technical, *mechanism):
        assert key in source
    assert "all(technical.values())" in source
    assert "all(mechanism.values())" in source
