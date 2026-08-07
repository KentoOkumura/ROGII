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
    numba_stub.get_num_threads = lambda: 1
    numba_stub.set_num_threads = lambda threads: None
    numba_stub.__version__ = "test-stub"
    sys.modules["numba"] = numba_stub

ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "experiments" / "exp440_ambiguity_gated_predictive_prior_hmm"
TRAIN_SOURCE = (
    EXP_DIR
    / "exp440_ambiguity_gated_predictive_prior_hmm_compact_selfcontained_train.py"
)
INFERENCE_SOURCE = (
    EXP_DIR
    / "exp440_ambiguity_gated_predictive_prior_hmm_compact_selfcontained_inference.py"
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
    EXP_DIR / "exp440_ambiguity_gated_predictive_prior_hmm_train.ipynb"
)
CANONICAL_INFERENCE = (
    EXP_DIR / "exp440_ambiguity_gated_predictive_prior_hmm_inference.ipynb"
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
    return load_module(TRAIN_SOURCE, "exp440_train_test")


@pytest.fixture(scope="module")
def inference():
    return load_module(INFERENCE_SOURCE, "exp440_inference_test")


@pytest.fixture(scope="module")
def exp209_reference():
    return load_module(EXP209_REFERENCE_SOURCE, "exp440_exp209_reference_test")


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
            "GR": 65.0
            + 8.0 * np.sin((typewell_tvt - 12_000.0) / 18.0),
        }
    )
    return horizontal, typewell


def reference_detector(
    values: np.ndarray,
    step: float,
    contract: dict,
) -> tuple:
    probabilities = np.asarray(values, dtype=np.float64)
    total = float(probabilities.sum())
    if not np.isfinite(total) or total <= 0.0:
        return False, 0, -1, -1, -1, 0.0, 0.0, 0.0, 0.0, 0.0
    probabilities = probabilities / total
    minimum = float(contract["min_peak_height"])
    peaks: list[int] = []
    if len(probabilities) == 1:
        peaks = [0] if probabilities[0] >= minimum else []
    elif len(probabilities) > 1:
        if probabilities[0] >= probabilities[1] and probabilities[0] >= minimum:
            peaks.append(0)
        for index in range(1, len(probabilities) - 1):
            if (
                probabilities[index] >= minimum
                and probabilities[index] >= probabilities[index - 1]
                and probabilities[index] > probabilities[index + 1]
            ):
                peaks.append(index)
        if (
            probabilities[-1] > probabilities[-2]
            and probabilities[-1] >= minimum
        ):
            peaks.append(len(probabilities) - 1)
    ranked = sorted(peaks, key=lambda index: (-float(probabilities[index]), index))
    if len(ranked) < 2:
        top1 = ranked[0] if ranked else -1
        return False, len(peaks), top1, -1, -1, 0.0, 0.0, 0.0, 0.0, 0.0
    top1, top2 = ranked[:2]
    low, high = sorted((top1, top2))
    valley = low + int(np.argmin(probabilities[low : high + 1]))
    lower_mass = float(probabilities[: valley + 1].sum())
    upper_mass = float(probabilities[valley + 1 :].sum())
    if top1 <= valley:
        top1_mass, top2_mass = lower_mass, upper_mass
    else:
        top1_mass, top2_mass = upper_mass, lower_mass
    minimum_peak = float(min(probabilities[top1], probabilities[top2]))
    valley_depth = (
        0.0
        if minimum_peak <= 0.0
        else 1.0 - float(probabilities[valley]) / minimum_peak
    )
    ratio = 0.0 if top1_mass <= 0.0 else top2_mass / top1_mass
    separation = abs(top1 - top2) * step
    active = (
        top2_mass >= float(contract["min_top2_mass"])
        and ratio >= float(contract["min_top2_to_top1_mass_ratio"])
        and separation >= float(contract["min_peak_separation_ft"])
        and valley_depth >= float(contract["min_valley_depth"])
    )
    return (
        active,
        len(peaks),
        top1,
        top2,
        valley,
        top1_mass,
        top2_mass,
        ratio,
        separation,
        valley_depth,
    )


def test_stage1_full_oof_completed_fail_closed_without_rerun(
    train, config, monkeypatch
):
    observed = train.validate_execution_contract(
        config, require_run_authorization=False
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
    assert config["experiment"]["status"] == "stage1_full_oof_failed_closed"
    assert config["execution"]["implementation_authorized"] is True
    assert config["execution"]["canonical_notebook_adoption_authorized"] is True
    assert config["execution"]["kaggle_package_authorized"] is True
    assert config["execution"]["stage0_run_authorized"] is True
    assert config["execution"]["rerun_authorized"] is False
    assert config["execution"]["stage1_run_authorized"] is True
    assert config["execution"]["stage1_prerequisite_override_authorized"] is True
    assert config["execution"]["stage1_rerun_authorized"] is False
    assert config["execution"]["inference_authorized"] is False
    assert config["execution"]["submission_authorized"] is False
    monkeypatch.setenv("EXP440_STAGE", "stage0_fixed32")
    with pytest.raises(RuntimeError, match="rerun is not authorized"):
        train.validate_execution_contract(
            config, require_run_authorization=True
        )
    monkeypatch.setenv("EXP440_STAGE", "stage1_merge")
    with pytest.raises(RuntimeError, match="Stage 1 failed closed"):
        train.validate_execution_contract(
            config, require_run_authorization=True
        )

    broken = deepcopy(config)
    broken["execution"]["stage1_rerun_authorized"] = True
    broken["execution"]["stage1_run_authorized"] = False
    with pytest.raises(RuntimeError, match="Stage 1 execution is not authorized"):
        train.validate_execution_contract(
            broken, require_run_authorization=True
        )


def test_scientific_contract_pins_parent_candidate_and_exp236(train, config):
    contract = train.validate_scientific_contract(config)
    assert contract["fixed_from_exp209"]["emission_family"] == (
        "gaussian_typewell_gr"
    )
    assert contract["candidate"]["ambiguous_emission_lambda"] == 0.0
    assert contract["candidate"]["clear_emission_lambda"] == 1.0
    assert contract["candidate"]["hard_previous_tvt_point_freeze"] is False
    assert contract["ambiguity_contract"] == {
        "source": "exp236_exact_hmm_posterior_bimodality_audit",
        "min_peak_height": 0.02,
        "min_top2_mass": 0.10,
        "min_top2_to_top1_mass_ratio": 0.25,
        "min_peak_separation_ft": 6.0,
        "min_valley_depth": 0.30,
    }

    broken = deepcopy(config)
    broken["model"]["ambiguity_contract"]["min_valley_depth"] = 0.29
    with pytest.raises(ValueError, match="exp236 bimodality contract"):
        train.validate_scientific_contract(broken)


def test_bimodality_detector_matches_exp236_reference(train, config):
    contract = config["model"]["ambiguity_contract"]
    step = config["model"]["fixed_from_exp209"]["position_grid_step_ft"]
    rng = np.random.default_rng(440)
    cases: list[np.ndarray] = []
    for _ in range(40):
        values = rng.gamma(shape=0.6, scale=1.0, size=61)
        values = np.convolve(values, np.asarray([0.2, 0.6, 0.2]), mode="same")
        cases.append(values)
    hand = np.full(61, 1.0e-6)
    hand[5] = 0.44
    hand[40] = 0.36
    hand[20] = 1.0e-8
    cases.append(hand)

    for values in cases:
        expected = reference_detector(values, step, contract)
        observed = train.bimodality_diagnostics_1d(
            values,
            step,
            contract["min_peak_height"],
            contract["min_top2_mass"],
            contract["min_top2_to_top1_mass_ratio"],
            contract["min_peak_separation_ft"],
            contract["min_valley_depth"],
        )
        assert tuple(observed[:5]) == tuple(expected[:5])
        np.testing.assert_allclose(
            np.asarray(observed[5:], dtype=np.float64),
            np.asarray(expected[5:], dtype=np.float64),
            rtol=0.0,
            atol=1.0e-12,
        )


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


def test_no_ambiguity_path_is_exact_parent_parity(train, config):
    parity = train.synthetic_no_ambiguity_parent_parity(
        config["model"]["fixed_from_exp209"],
        config["model"]["ambiguity_contract"],
    )
    assert parity["pass"] is True
    assert parity["active_rows"] == 0
    assert parity["posterior_mean_max_abs_diff_ft"] == 0.0
    assert parity["posterior_std_max_abs_diff_ft"] == 0.0
    assert (
        parity["parent_prediction_sha256"]
        == parity["candidate_prediction_sha256"]
    )


def test_gate_disabled_matches_independent_exp209_reference(
    train, exp209_reference, config
):
    fixed = config["model"]["fixed_from_exp209"]
    rows = 8
    positions = 27
    step = float(fixed["position_grid_step_ft"])
    grid = 11_900.0 + np.arange(positions, dtype=np.float64) * step
    rates = np.linspace(-0.10, 0.10, int(fixed["n_rates"]), dtype=np.float64)
    x = np.linspace(-1.0, 1.0, positions)
    emission = np.vstack(
        [
            -0.5 * ((x - 0.25 * np.sin(row / 2.0)) / 0.37) ** 2
            for row in range(rows)
        ]
    ).astype(np.float32)
    dm = np.linspace(8.0, 19.0, rows, dtype=np.float64)
    dz = np.linspace(-0.2, 0.5, rows, dtype=np.float64)
    prepared = {
        "emission_ll": emission,
        "raw_gr_missing": np.zeros(rows, dtype=bool),
        "dm": dm,
        "dz": dz,
        "grid": grid,
        "rates": rates,
        "start_p": 12.5,
        "r0": 0.0,
    }
    decoded = train.run_ambiguity_gated_hmm(
        prepared,
        fixed,
        config["model"]["ambiguity_contract"],
        gate_enabled=False,
    )
    reference_posterior, reference_loglik = exp209_reference._hmm2_fb(
        emission,
        dm,
        dz,
        step,
        rates,
        float(fixed["sig_r"]),
        float(fixed["sig_p"]),
        12.5,
        float(fixed["start_sigma_ft"]),
        0.0,
        float(fixed["initial_rate_sigma"]),
        1.0,
        float(fixed["momentum"]),
    )
    reference_mean = reference_posterior @ grid
    reference_std = np.sqrt(
        np.maximum(reference_posterior @ (grid**2) - reference_mean**2, 0.0)
    )
    np.testing.assert_allclose(
        decoded["posterior_mean"], reference_mean, rtol=0.0, atol=1.0e-10
    )
    np.testing.assert_allclose(
        decoded["posterior_std"], reference_std, rtol=0.0, atol=1.0e-10
    )
    assert decoded["log_likelihood"] == pytest.approx(
        reference_loglik, abs=1.0e-10
    )


def test_observed_only_gate_holds_predictive_and_freezes_backward_schedule(
    train, config
):
    fixed = config["model"]["fixed_from_exp209"]
    rows = 3
    positions = 31
    grid = 11_900.0 + np.arange(positions) * fixed["position_grid_step_ft"]
    rates = np.linspace(-0.10, 0.10, fixed["n_rates"])
    emission = np.full((rows, positions), -20.0, dtype=np.float32)
    emission[:, 13] = 0.0
    emission[:, 17] = 0.0
    permissive = {
        "min_peak_height": 1.0e-4,
        "min_top2_mass": 1.0e-3,
        "min_top2_to_top1_mass_ratio": 1.0e-3,
        "min_peak_separation_ft": 0.35,
        "min_valley_depth": 0.01,
    }
    prepared = {
        "emission_ll": emission,
        "raw_gr_missing": np.asarray([False, True, False]),
        "dm": np.full(rows, 10.0),
        "dz": np.zeros(rows),
        "grid": grid,
        "rates": rates,
        "start_p": 15.0,
        "r0": 0.0,
    }
    decoded = train.run_ambiguity_gated_hmm(
        prepared, fixed, permissive, gate_enabled=True
    )
    assert decoded["raw_bimodal"][0]
    assert decoded["ambiguity_active"][0]
    assert not decoded["ambiguity_active"][1]
    active = decoded["ambiguity_active"]
    np.testing.assert_allclose(
        decoded["candidate_filtered_mean"][active],
        decoded["predictive_mean"][active],
        rtol=0.0,
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        decoded["candidate_filtered_mean"][~active],
        decoded["provisional_mean"][~active],
        rtol=0.0,
        atol=1.0e-10,
    )
    assert decoded["schedule_sha256"] == decoded["backward_schedule_sha256"]


def test_leakage_ledger_blocks_role_truth_episode_and_cause_until_all_freeze(train):
    ledger = train.LeakageLedger(expected_wells=2)
    ledger.freeze(
        "a",
        schedule_sha256="s1",
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
        schedule_sha256="s2",
        prediction_sha256="p2",
        diagnostic_sha256="d2",
    )
    ledger.record_role_fold_late(2)
    ledger.record_truth_late(5)
    ledger.record_episode_late(1)
    ledger.record_cause_late(1)
    assert ledger.all_frozen


def test_fixed32_manifest_is_sha_pinned_and_late_identity_is_balanced(config):
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
            "predictive_mean": [12_000.125, 12_000.375],
            "ambiguity_active": [True, False],
        }
    )
    report = train.write_deterministic_gzip_csv(
        tmp_path / "round_trip.csv.gz", frame
    )
    assert report["logical_sha256"] == report["readback_logical_sha256"]
    assert report["logical_sha256"] == report["decompressed_sha256"]


def test_inference_remains_fail_closed(inference, config):
    contract = inference.validate_inference_disabled(config)
    assert contract["implementation_authorized"] is True
    assert contract["canonical_notebook_adoption_authorized"] is True
    assert contract["kaggle_package_authorized"] is True
    assert contract["stage0_run_authorized"] is True
    assert contract["stage1_run_authorized"] is True
    assert contract["inference_authorized"] is False
    assert contract["submission_authorized"] is False
    assert contract["create_submission"] is False
    with pytest.raises(RuntimeError, match="inference is disabled"):
        inference.run_inference(config)


def test_compact_candidates_are_self_contained_and_train_is_canonical():
    train_source = TRAIN_SOURCE.read_text()
    inference_source = INFERENCE_SOURCE.read_text()
    assert "__file__" not in train_source
    assert "__file__" not in inference_source
    assert "from exact_hmm_smoother import" not in train_source
    assert "from posterior_bimodality_audit import" not in train_source
    assert "bimodality_diagnostics_1d" in train_source
    assert "_hmm2_ambiguity_gated" in train_source
    assert CANONICAL_TRAIN.is_file()
    assert CANONICAL_INFERENCE.is_file()
    assert "bimodality_diagnostics_1d" in CANONICAL_TRAIN.read_text()
    assert "_hmm2_ambiguity_gated" in CANONICAL_TRAIN.read_text()
    assert "bimodality_diagnostics_1d" not in CANONICAL_INFERENCE.read_text()


def test_all_stage0_gate_keys_are_consumed_by_one_and_gate(train, config):
    source = inspect.getsource(train.evaluate_stage0_gates)
    technical = config["gates"]["stage0_fixed32"]["technical"]
    mechanism = config["gates"]["stage0_fixed32"]["mechanism"]
    for key in technical:
        assert key in source
    for key in mechanism:
        assert key in source
    assert "all(technical.values()) and all(mechanism.values())" in source
    assert config["gates"]["stage0_fixed32"]["fail_action"] in CONFIG_PATH.read_text()


def test_stage1_lpt_assigns_every_well_once_and_balances_rows(train):
    manifest = pd.DataFrame(
        {
            "well": [f"w{index}" for index in range(12)],
            "suffix_rows": [120, 100, 80, 70, 60, 50, 40, 30, 20, 10, 8, 4],
        }
    )
    first = train.assign_stage1_lpt_shards(manifest)
    second = train.assign_stage1_lpt_shards(manifest.sample(frac=1.0, random_state=440))
    first_map = first.set_index("well")["shard_index"].sort_index()
    second_map = second.set_index("well")["shard_index"].sort_index()
    pd.testing.assert_series_equal(first_map, second_map)
    assert sorted(first["shard_index"].unique().tolist()) == [0, 1, 2, 3]
    assert first["well"].nunique() == len(manifest)
    loads = first.groupby("shard_index")["suffix_rows"].sum()
    assert int(loads.max() - loads.min()) <= int(manifest["suffix_rows"].max())


def test_stage1_metric_scopes_use_frozen_nonworse_contract(train, config):
    rows = []
    for well_index in range(10):
        for row_index in range(3):
            truth = float(100 + well_index + row_index)
            rows.append(
                {
                    "well": f"w{well_index}",
                    "row_idx": row_index,
                    "fold": well_index % 5,
                    "tvt_true": truth,
                    "parent_prediction": truth + 2.0,
                    "candidate_prediction": truth + 1.0,
                    "raw_gr_observed": row_index != 1,
                    "raw_gr_missing": row_index == 1,
                    "well_missing_fraction": 1.0 / 3.0,
                    "md_since": float(row_index * 1000),
                    "hidden_like_spatial": well_index < 5,
                    "hidden_like_typewell_purged": well_index >= 5,
                    "ambiguity_active": row_index == 0,
                }
            )
    frame = pd.DataFrame(rows)
    metrics, by_well = train.build_stage1_metrics(config, frame)
    expected_scopes = {
        "overall",
        "fold_0",
        "fold_1",
        "fold_2",
        "fold_3",
        "fold_4",
        "raw_gr_observed",
        "raw_gr_missing",
        "high_missing_fraction",
        "md_1000_plus",
        "hidden_like_spatial",
        "hidden_like_typewell_purged",
    }
    assert set(metrics["scope"]) == expected_scopes
    assert (metrics["improvement_ft"] > 0.0).all()
    assert (
        by_well["rmse_delta_candidate_minus_parent_ft"] < 0.0
    ).all()
