from __future__ import annotations

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
    numba_stub.get_num_threads = lambda: 1
    numba_stub.set_num_threads = lambda threads: None
    numba_stub.__version__ = "test-stub"
    sys.modules["numba"] = numba_stub

ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "experiments" / "exp482_isolated_gr_shock_prior_hold"
TRAIN_SOURCE = EXP_DIR / "exp482_isolated_gr_shock_prior_hold_compact_selfcontained_train.py"
INFERENCE_SOURCE = (
    EXP_DIR / "exp482_isolated_gr_shock_prior_hold_compact_selfcontained_inference.py"
)
CONFIG_PATH = EXP_DIR / "config.yaml"
CANONICAL_TRAIN = EXP_DIR / "exp482_isolated_gr_shock_prior_hold_train.ipynb"
CANONICAL_INFERENCE = EXP_DIR / "exp482_isolated_gr_shock_prior_hold_inference.ipynb"
EXP209_REFERENCE_SOURCE = (
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
    return load_module(TRAIN_SOURCE, "exp482_train_test")


@pytest.fixture(scope="module")
def inference():
    return load_module(INFERENCE_SOURCE, "exp482_inference_test")


@pytest.fixture(scope="module")
def exp209_reference():
    return load_module(EXP209_REFERENCE_SOURCE, "exp482_exp209_reference_test")


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


def test_stage0_execution_contract_is_authorized_but_stage1_is_not(train, config):
    observed = train.validate_execution_contract(config, require_run_authorization=False)
    assert observed == {
        "scientific_variants": 1,
        "stage0_raw_census_wells": 773,
        "stage0_parent_message_hmm_replays": 64,
        "stage1_parent_message_hmm_replays": 773,
        "candidate_state_modifying_hmm_runs": 0,
        "saved_parent_prediction_reruns": 0,
        "lightgbm_configs": 0,
        "trained_ml_folds": 0,
        "boosters": 0,
        "fitted_models": 0,
        "pf_runs": 0,
        "beam_runs": 0,
        "gpu_runs": 0,
    }
    assert config["experiment"]["status"] == "stage_a0_eligibility_failed_closed"
    assert config["execution"]["implementation_authorized"] is True
    assert config["execution"]["canonical_notebook_adoption_authorized"] is True
    assert config["execution"]["kaggle_package_authorized"] is True
    assert config["execution"]["stage0_run_authorized"] is True
    assert config["execution"]["stage1_run_authorized"] is False
    assert config["execution"]["inference_authorized"] is False
    assert config["execution"]["submission_authorized"] is False
    assert train.validate_execution_contract(config, require_run_authorization=True) == observed


def test_scientific_contract_is_pinned(train, config):
    contract = train.validate_scientific_contract(config)
    assert contract["fixed_from_exp209"]["position_grid_step_ft"] == 0.35
    assert contract["fixed_from_exp209"]["n_rates"] == 41
    assert contract["raw_shock"]["robust_z_min"] == 4.5
    assert contract["message_agreement"] == {
        "predictive_mean_vs_loo_mean_max_ft": 1.05,
        "predictive_or_loo_std_max_ft": 6.0,
    }
    assert contract["current_emission_conflict"] == {
        "predictive_to_provisional_mean_shift_min_ft": 1.05,
        "saved_parent_to_loo_output_difference_min_ft": 0.35,
    }
    broken = deepcopy(config)
    broken["model"]["raw_shock"]["robust_z_min"] = 4.4
    with pytest.raises(ValueError, match="raw-shock contract"):
        train.validate_scientific_contract(broken)


def test_single_raw_gr_shock_and_cluster_suppression(train, config):
    raw = np.full(25, 50.0)
    raw[12] = 100.0
    diagnostics = train.isolated_raw_shock_diagnostics(raw, config["model"]["raw_shock"])
    assert diagnostics.loc[12, "robust_z"] == pytest.approx(50.0)
    assert diagnostics.loc[12, "raw_shock_precluster"]
    assert diagnostics.loc[12, "isolated_raw_shock"]
    assert not diagnostics.loc[:4, "isolated_raw_shock"].any()
    assert not diagnostics.loc[20:, "isolated_raw_shock"].any()

    clustered = raw.copy()
    clustered[14] = 100.0
    clustered_diagnostics = train.isolated_raw_shock_diagnostics(
        clustered, config["model"]["raw_shock"]
    )
    assert clustered_diagnostics.loc[12, "raw_shock_precluster"]
    assert clustered_diagnostics.loc[14, "raw_shock_precluster"]
    assert not clustered_diagnostics.loc[12, "isolated_raw_shock"]
    assert not clustered_diagnostics.loc[14, "isolated_raw_shock"]


def test_raw_shock_requires_three_finite_neighbors_per_side(train, config):
    raw = np.full(21, np.nan)
    raw[10] = 100.0
    raw[[5, 6, 9, 11, 14, 15]] = 50.0
    diagnostics = train.isolated_raw_shock_diagnostics(raw, config["model"]["raw_shock"])
    assert diagnostics.loc[10, "finite_left_neighbors"] == 3
    assert diagnostics.loc[10, "finite_right_neighbors"] == 3
    assert diagnostics.loc[10, "isolated_raw_shock"]
    raw[5] = np.nan
    diagnostics = train.isolated_raw_shock_diagnostics(raw, config["model"]["raw_shock"])
    assert diagnostics.loc[10, "finite_left_neighbors"] == 2
    assert not diagnostics.loc[10, "raw_shock_precluster"]


def synthetic_census() -> pd.DataFrame:
    rows = []
    for index in range(80):
        shock_count = 80 - index if index < 40 else 0
        rows.append(
            {
                "well": f"w{index:03d}",
                "suffix_rows": 1_000 + 7 * index,
                "raw_missing_fraction": (index % 11) / 20.0,
                "raw_shock_precluster_count": shock_count,
                "shock_count": shock_count,
                "horizontal_raw_sha256": f"{index:064x}",
            }
        )
    return pd.DataFrame(rows)


def test_fixed64_manifest_is_target_free_unique_and_order_invariant(train, config):
    census = synthetic_census()
    first = train.build_fixed64_manifest(config, census)
    second = train.build_fixed64_manifest(config, census.sample(frac=1.0, random_state=482))
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 64
    assert first["well"].nunique() == 64
    assert first["selection_role"].value_counts().to_dict() == {
        "support": 32,
        "control": 32,
    }
    assert (first.loc[first["selection_role"].eq("control"), "shock_count"] == 0).all()
    assert set(first.columns).isdisjoint(
        {"TVT", "tvt_true", "fold", "error", "hidden_like_spatial"}
    )


def test_raw_census_eligibility_is_an_and_gate(train, config):
    census = synthetic_census()
    eligibility = train.raw_census_eligibility(config, census)
    assert eligibility["passed"] is True
    broken = census.copy()
    broken["shock_count"] = 0
    eligibility = train.raw_census_eligibility(config, broken)
    assert eligibility["passed"] is False
    assert eligibility["checks"]["minimum_isolated_raw_shock_rows"] is False
    assert eligibility["checks"]["minimum_support_wells"] is False


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
            np.nanstd(known["GR"].fillna(0).to_numpy(np.float64) - typewell_at_known),
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


def test_parent_and_leave_one_out_messages_match_independent_reruns(
    train, exp209_reference, config
):
    fixed = config["model"]["fixed_from_exp209"]
    rows = 6
    positions = 25
    step = float(fixed["position_grid_step_ft"])
    rates = np.linspace(-0.10, 0.10, int(fixed["n_rates"]))
    x = np.linspace(-1.0, 1.0, positions)
    emission = np.vstack(
        [-0.5 * ((x - 0.30 * np.sin(row / 2.0)) / 0.35) ** 2 for row in range(rows)]
    ).astype(np.float32)
    dm = np.linspace(8.0, 16.0, rows)
    dz = np.linspace(-0.15, 0.30, rows)
    observed = train._hmm2_parent_and_loo_position_marginals(
        emission,
        dm,
        dz,
        step,
        rates,
        float(fixed["rate_process_sigma"]),
        float(fixed["position_process_sigma"]),
        12.0,
        float(fixed["start_sigma_ft"]),
        0.0,
        float(fixed["initial_rate_sigma"]),
        float(fixed["emission_lambda"]),
        float(fixed["momentum"]),
    )
    parent, predictive, provisional, loo = observed
    reference_parent, _ = exp209_reference._hmm2_fb(
        emission,
        dm,
        dz,
        step,
        rates,
        float(fixed["rate_process_sigma"]),
        float(fixed["position_process_sigma"]),
        12.0,
        float(fixed["start_sigma_ft"]),
        0.0,
        float(fixed["initial_rate_sigma"]),
        float(fixed["emission_lambda"]),
        float(fixed["momentum"]),
    )
    np.testing.assert_allclose(parent, reference_parent, rtol=0.0, atol=5.0e-7)
    for row in range(rows):
        emission_without_current = emission.copy()
        emission_without_current[row] = 0.0
        reference_loo, _ = exp209_reference._hmm2_fb(
            emission_without_current,
            dm,
            dz,
            step,
            rates,
            float(fixed["rate_process_sigma"]),
            float(fixed["position_process_sigma"]),
            12.0,
            float(fixed["start_sigma_ft"]),
            0.0,
            float(fixed["initial_rate_sigma"]),
            float(fixed["emission_lambda"]),
            float(fixed["momentum"]),
        )
        np.testing.assert_allclose(loo[row], reference_loo[row], rtol=0.0, atol=5.0e-7)
    np.testing.assert_allclose(loo[-1], predictive[-1], rtol=0.0, atol=5.0e-7)
    assert np.isfinite(provisional).all()


def test_trigger_is_strict_and_and_row_local(train, config):
    rows = 4
    raw = pd.DataFrame(
        {
            "suffix_offset": np.arange(rows),
            "raw_gr_observed": True,
            "raw_shock_precluster": [False, True, False, False],
            "isolated_raw_shock": [False, True, False, False],
            "robust_z": [0.0, 8.0, 0.0, 0.0],
        }
    )
    summaries = {
        "predictive": {
            "mean": np.full(rows, 100.0),
            "std": np.full(rows, 1.0),
        },
        "provisional": {
            "mean": np.asarray([100.0, 102.0, 100.0, 100.0]),
            "std": np.full(rows, 1.0),
        },
        "parent": {
            "mean": np.asarray([100.0, 101.0, 100.0, 100.0]),
            "std": np.full(rows, 1.0),
        },
        "loo": {
            "mean": np.asarray([100.0, 100.5, 100.0, 100.0]),
            "std": np.full(rows, 1.0),
        },
    }
    saved_parent = np.asarray([90.0, 101.0, 92.0, 93.0])
    trigger = train.build_isolated_shock_trigger(config, raw, summaries, saved_parent)
    assert trigger["trigger_active"].tolist() == [False, True, False, False]
    assert trigger["candidate_prediction"].tolist() == [90.0, 100.5, 92.0, 93.0]
    assert trigger.loc[2, "candidate_prediction"] == saved_parent[2]

    too_uncertain = deepcopy(summaries)
    too_uncertain["loo"]["std"] = np.asarray([1.0, 6.01, 1.0, 1.0])
    trigger = train.build_isolated_shock_trigger(config, raw, too_uncertain, saved_parent)
    assert not trigger["trigger_active"].any()


def test_leakage_ledger_requires_every_target_free_freeze(train):
    ledger = train.LeakageLedger(expected_wells=2)
    with pytest.raises(RuntimeError, match="truth"):
        ledger.record_truth_late(3)
    ledger.freeze_census("census", "shock")
    ledger.freeze_manifest("manifest")
    ledger.freeze_well(
        "a",
        message_sha256="message-a",
        trigger_sha256="trigger-a",
        prediction_sha256="prediction-a",
    )
    with pytest.raises(RuntimeError, match="fold"):
        ledger.record_fold_late(2)
    ledger.freeze_well(
        "b",
        message_sha256="message-b",
        trigger_sha256="trigger-b",
        prediction_sha256="prediction-b",
    )
    ledger.record_truth_late(10)
    ledger.record_fold_late(10)
    assert ledger.all_frozen
    assert ledger.forbidden_rows_before_all_freeze == 5


def test_all_stage0_gate_keys_are_consumed_by_one_and_gate(train, config):
    source = inspect.getsource(train.evaluate_stage0_gates)
    technical = config["validation"]["stage0"]["technical"]
    scientific = config["validation"]["stage0"]["scientific"]
    for key in technical:
        assert key in source
    for key in scientific:
        assert key in source
    assert "all(technical.values())" in source
    assert config["validation"]["stage0"]["fail_action"] in CONFIG_PATH.read_text()


def test_deterministic_gzip_round_trip_sha(train, tmp_path):
    frame = pd.DataFrame(
        {
            "well": ["a", "a"],
            "row_idx": [1, 2],
            "loo_mean": [12_000.125, 12_000.375],
            "trigger_active": [True, False],
        }
    )
    report = train.write_deterministic_gzip_csv(tmp_path / "round_trip.csv.gz", frame)
    assert report["logical_sha256"] == report["readback_logical_sha256"]
    assert report["logical_sha256"] == report["decompressed_sha256"]


def test_inference_remains_fail_closed(inference, config):
    contract = inference.validate_inference_disabled(config)
    assert contract == {
        "implementation_authorized": True,
        "canonical_notebook_adoption_authorized": True,
        "kaggle_package_authorized": True,
        "stage0_run_authorized": True,
        "stage1_run_authorized": False,
        "inference_authorized": False,
        "submission_authorized": False,
        "create_submission": False,
    }
    with pytest.raises(RuntimeError, match="inference is disabled"):
        inference.run_inference(config)


def test_compact_sources_are_self_contained_and_train_is_canonical():
    train_source = TRAIN_SOURCE.read_text()
    inference_source = INFERENCE_SOURCE.read_text()
    assert "__file__" not in train_source
    assert "__file__" not in inference_source
    assert "from exact_hmm_smoother import" not in train_source
    assert "from settings import" not in train_source
    assert "_hmm2_parent_and_loo_position_marginals" in train_source
    assert "isolated_raw_shock_diagnostics" in train_source
    assert CANONICAL_TRAIN.is_file()
    assert CANONICAL_INFERENCE.is_file()
    canonical_train = CANONICAL_TRAIN.read_text()
    assert "_hmm2_parent_and_loo_position_marginals" in canonical_train
    assert "isolated_raw_shock_diagnostics" in canonical_train
    assert "run_inference" not in CANONICAL_INFERENCE.read_text()
