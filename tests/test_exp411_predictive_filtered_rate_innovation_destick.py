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
    sys.modules["numba"] = numba_stub

ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = (
    ROOT
    / "experiments"
    / "exp411_predictive_filtered_rate_innovation_destick"
)
TRAIN_SOURCE = (
    EXP_DIR
    / "exp411_predictive_filtered_rate_innovation_destick_compact_selfcontained_train.py"
)
INFERENCE_SOURCE = (
    EXP_DIR
    / "exp411_predictive_filtered_rate_innovation_destick_"
    "compact_selfcontained_inference.py"
)
CONFIG_PATH = EXP_DIR / "config.yaml"
MANIFEST_PATH = EXP_DIR / "assets" / "stage0_fixed32_manifest.csv"
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
    return load_module(TRAIN_SOURCE, "exp411_train_test")


@pytest.fixture(scope="module")
def inference():
    return load_module(INFERENCE_SOURCE, "exp411_inference_test")


@pytest.fixture(scope="module")
def parent():
    return load_module(PARENT_SOURCE, "exp411_exp209_reference")


@pytest.fixture(scope="module")
def config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())


def test_stage0_execution_contract_is_authorized_and_fixed32_only(
    train,
    config,
):
    counts = train.validate_execution_contract(
        config,
        require_run_authorization=False,
    )
    assert counts == {
        "active_treatment_variants": 1,
        "stage_0_hmm_well_runs": 32,
        "parent_control_hmm_reruns": 0,
        "lightgbm_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
        "models": 0,
        "pf_well_runs": 0,
        "beam_well_runs": 0,
        "gpu_runs": 0,
    }
    assert str(config["experiment"]["status"]).startswith("stage0_")
    assert config["design"]["implementation_enabled"] is True
    assert config["design"]["stage_0_execution_approved"] is True
    assert config["design"]["stage_1_execution_approved"] is False
    assert config["execution"]["kaggle_execution_authorized"] is True
    assert train.validate_execution_contract(
        config,
        require_run_authorization=True,
    ) == counts


def test_contract_rejects_stage1_inference_or_parent_rerun(train, config):
    broken = deepcopy(config)
    broken["design"]["stage_1_execution_approved"] = True
    with pytest.raises(ValueError, match="Stage 1"):
        train.validate_execution_contract(broken, require_run_authorization=False)

    broken = deepcopy(config)
    broken["design"]["inference_enabled"] = True
    with pytest.raises(ValueError, match="inference"):
        train.validate_execution_contract(broken, require_run_authorization=False)

    broken = deepcopy(config)
    broken["execution"]["parent_control_hmm_reruns"] = 1
    with pytest.raises(ValueError, match="execution contract"):
        train.validate_execution_contract(broken, require_run_authorization=False)


def test_scientific_contract_pins_parent_trigger_and_single_change(train, config):
    contract = train.validate_scientific_contract(config)
    assert contract["parent_hmm"]["sig_r"] == 0.002
    assert contract["parent_hmm"]["sig_p"] == 0.02
    assert contract["parent_hmm"]["mom"] == 0.998
    assert contract["trigger"]["activation_transitions"] == 32
    assert contract["trigger"]["refractory_rows"] == 128
    assert contract["trigger"]["cusum_updates_during_active_and_refractory"] is True
    assert contract["treatment"]["stay_mass_transfer_fraction"] == 0.10
    assert contract["treatment"]["first_affected_transition"] == "row_after_trigger"

    broken = deepcopy(config)
    broken["model"]["trigger"]["positive_threshold_rate_cells"] = 0.9
    with pytest.raises(ValueError, match="trigger contract changed"):
        train.validate_scientific_contract(broken)


def test_saved_parent_prediction_sha_matches_exp209_anchor(config):
    expected = (
        "8e2f42367b7b8b28e73094eae642c57c75dc8a7ebcfbc3826b0f2067b37f7ae5"
    )
    assert len(expected) == 64
    assert config["data"]["parent_hmm_cache_decompressed_sha256"] == expected
    assert (
        config["data"]["exp209_saved_control"]["expected_decompressed_sha256"]
        == expected
    )


def test_parent_row_index_is_parsed_from_exact_well_prefixed_cache_id(train, parent):
    assert parent.META_COLUMNS == ["id", "well", "target"]
    assert "row_idx" not in parent.META_COLUMNS
    frame = pd.DataFrame(
        {
            "well": ["alpha_beta", "gamma"],
            "id": ["alpha_beta_17", "gamma_2048"],
        }
    )
    np.testing.assert_array_equal(
        train.parent_row_indices_from_cache_ids(frame),
        np.asarray([17, 2048], dtype=np.int64),
    )

    with pytest.raises(ValueError, match="exact well prefix"):
        train.parent_row_indices_from_cache_ids(
            pd.DataFrame({"well": ["alpha"], "id": ["beta_17"]})
        )
    with pytest.raises(ValueError, match="invalid row suffix"):
        train.parent_row_indices_from_cache_ids(
            pd.DataFrame({"well": ["alpha"], "id": ["alpha_not-a-row"]})
        )
    np.testing.assert_array_equal(
        train.parent_cache_ids_for_rows(
            "alpha_beta",
            np.asarray([17, 2048], dtype=np.int64),
        ),
        np.asarray(["alpha_beta_17", "alpha_beta_2048"]),
    )


def test_hmm_input_preparation_uses_raw_range_index_without_id_column(train, config):
    horizontal = pd.DataFrame(
        {
            "MD": np.arange(8, dtype=float) * 10.0,
            "Z": np.linspace(0.0, 0.7, 8),
            "GR": np.linspace(60.0, 67.0, 8),
            "TVT_input": [1000.0, 1000.2, 1000.4, 1000.6]
            + [np.nan] * 4,
        }
    )
    typewell = pd.DataFrame(
        {
            "TVT": np.linspace(900.0, 1100.0, 64),
            "GR": np.linspace(50.0, 80.0, 64),
        }
    )
    prepared = train.prepare_hmm_inputs(
        horizontal,
        typewell,
        config["model"]["parent_hmm"],
    )
    assert "eval_id" not in prepared
    np.testing.assert_array_equal(
        prepared["eval_index"],
        np.asarray([4, 5, 6, 7], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        train.parent_cache_ids_for_rows("well_with_underscore", prepared["eval_index"]),
        np.asarray(
            [
                "well_with_underscore_4",
                "well_with_underscore_5",
                "well_with_underscore_6",
                "well_with_underscore_7",
            ]
        ),
    )


def test_truth_late_loader_reconstructs_ids_after_all_wells_freeze(train, tmp_path):
    well = "well_with_underscore"
    pd.DataFrame(
        {
            "MD": np.arange(6, dtype=float) * 10.0,
            "Z": np.linspace(0.0, 0.5, 6),
            "TVT": np.linspace(1000.0, 1001.0, 6),
            "TVT_input": [1000.0, 1000.2, 1000.4, np.nan, np.nan, np.nan],
        }
    ).to_csv(tmp_path / f"{well}__horizontal_well.csv", index=False)
    frozen = types.SimpleNamespace(
        well=well,
        row_idx=np.asarray([3, 4, 5], dtype=np.int64),
        eval_id=np.asarray(
            [
                "well_with_underscore_3",
                "well_with_underscore_4",
                "well_with_underscore_5",
            ]
        ),
    )
    ledger = train.LeakageLedger(expected_wells=1)
    ledger.freeze(well)
    truth = train.load_truth_after_all_freeze(frozen, tmp_path, ledger)
    assert truth["id"].tolist() == frozen.eval_id.tolist()
    assert truth["row_idx"].tolist() == [3, 4, 5]
    assert ledger.truth_rows_before_all_freeze == 0
    assert ledger.truth_rows_after_all_freeze == 3


def test_deterministic_gzip_readback_preserves_float_round_trip_sha(train, tmp_path):
    frame = pd.DataFrame(
        {
            "well": ["060ab2b8", "060ab2b8"],
            "row_idx": [1560, 1561],
            "predictive_rate_mean": [
                0.01996000006095223,
                0.01991804125470167,
            ],
            "filtered_rate_mean": [
                0.019957957158047136,
                0.019915259059984376,
            ],
            "innovation_rate_cells": [
                -0.00040858058101903016,
                -0.0005564389434588313,
            ],
        }
    )
    report = train.write_deterministic_gzip_csv(
        tmp_path / "round_trip.csv.gz",
        frame,
    )
    assert report["logical_sha256"] == report["readback_logical_sha256"]
    assert report["logical_sha256"] == report["decompressed_sha256"]


def test_fixed32_manifest_is_sha_pinned_balanced_and_unique(config):
    observed = hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest()
    assert observed == config["data"]["stage_0_manifest"]["expected_sha256"]
    manifest = pd.read_csv(MANIFEST_PATH, dtype={"well": str})
    assert len(manifest) == 32
    assert manifest["well"].nunique() == 32
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
    assert set(manifest["quartile_match_distance"].astype(int)) <= {0, 1, 2}
    assert (
        manifest.loc[manifest["role"].eq("persistent"), "quartile_match_distance"]
        .eq(0)
        .all()
    )


def test_directional_rate_kernel_moves_only_stay_mass_and_keeps_edge_noop(train):
    rates = np.linspace(-0.10, 0.10, 41)
    base = train.rate_kernel_probabilities(rates, 12.0, 0.002, 0.998, 0, 0.10)
    positive = train.rate_kernel_probabilities(
        rates, 12.0, 0.002, 0.998, 1, 0.10
    )
    negative = train.rate_kernel_probabilities(
        rates, 12.0, 0.002, 0.998, -1, 0.10
    )
    interior = 20
    moved = 0.10 * base[interior, 1]
    assert positive[interior, 2] == pytest.approx(base[interior, 2] + moved)
    assert positive[interior, 1] == pytest.approx(base[interior, 1] - moved)
    assert positive[interior, 0] == pytest.approx(base[interior, 0])
    assert negative[interior, 0] == pytest.approx(base[interior, 0] + moved)
    assert negative[interior, 1] == pytest.approx(base[interior, 1] - moved)
    assert negative[interior, 2] == pytest.approx(base[interior, 2])
    np.testing.assert_allclose(positive[-1], base[-1], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(negative[0], base[0], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(positive.sum(axis=1), base.sum(axis=1), atol=1e-14)
    np.testing.assert_allclose(negative.sum(axis=1), base.sum(axis=1), atol=1e-14)


def test_trigger_affects_next_32_transitions_then_enforces_128_row_refractory(train):
    innovation = np.zeros(170, dtype=np.float64)
    innovation[0] = 1.25
    innovation[40:161] = 0.05
    triggers, active = train.cusum_activation_schedule_from_innovation(
        innovation,
        drift_allowance=0.01,
        positive_threshold=1.0,
        negative_threshold=1.0,
        tie_tolerance=1e-12,
        activation_transitions=32,
        refractory_rows=128,
    )
    assert triggers[0] == 1
    assert not np.any(active[:1])
    assert np.all(active[1:33] == 1)
    assert not np.any(active[33:161])
    assert not np.any(triggers[1:161])
    assert triggers[161] == 1
    assert active[161] == 0
    assert np.all(active[162:] == 1)


def test_no_trigger_small_trellis_matches_parent_to_exact_tolerance(train, config):
    parity = train.synthetic_no_trigger_parent_parity(
        config["model"]["parent_hmm"],
        config["model"]["trigger"],
        config["model"]["treatment"],
    )
    assert parity["pass"] is True
    assert parity["posterior_mean_max_abs_diff_ft"] <= 1e-10
    assert parity["log_likelihood_abs_diff"] <= 1e-10
    assert parity["active_rows"] == 0
    assert parity["trigger_rows"] == 0


def test_untreated_kernel_matches_independent_exp209_reference(train, parent, config):
    rng = np.random.default_rng(411)
    row_count = 32
    emission = rng.normal(0.0, 0.3, size=(row_count, 13)).astype(np.float32)
    dm = 1.0 + (np.arange(row_count, dtype=np.float64) % 5) * 0.2
    dz = 0.25 * np.sin(np.arange(row_count, dtype=np.float64) / 7.0)
    rates = np.linspace(-0.06, 0.06, 9, dtype=np.float64)
    common = (
        dm,
        dz,
        0.35,
        rates,
        0.002,
        0.02,
        6.0,
        0.75,
        0.01,
        0.01,
        1.0,
        0.998,
    )
    reference_position, reference_loglik = parent._hmm2_fb(emission, *common)
    trigger = config["model"]["trigger"]
    treatment = config["model"]["treatment"]
    observed = train._hmm2_directional_destick(
        emission,
        *common,
        trigger["innovation_scale_rate_step"],
        trigger["drift_allowance_rate_cells"],
        trigger["positive_threshold_rate_cells"],
        trigger["negative_threshold_rate_cells"],
        trigger["tie_tolerance"],
        trigger["activation_transitions"],
        trigger["refractory_rows"],
        treatment["stay_mass_transfer_fraction"],
        False,
    )
    np.testing.assert_allclose(observed[0], reference_position, rtol=0.0, atol=2e-7)
    assert abs(float(observed[1]) - float(reference_loglik)) <= 2e-6
    assert not np.any(observed[7])
    assert not np.any(observed[8])


def test_forward_schedule_oracle_matches_kernel_output(train, config):
    hmm = config["model"]["parent_hmm"]
    trigger = config["model"]["trigger"]
    treatment = config["model"]["treatment"]
    rows = 20
    positions = 15
    prepared = {
        "emission_ll": np.vstack(
            [
                -0.5
                * (
                    (
                        np.linspace(-1.0, 1.0, positions)
                        - 0.6 * np.sin(index / 2.0)
                    )
                    / 0.35
                )
                ** 2
                for index in range(rows)
            ]
        ).astype(np.float32),
        "dm": np.full(rows, 15.0),
        "dz": np.linspace(-0.2, 0.4, rows),
        "grid": 11_900.0 + np.arange(positions) * 0.35,
        "rates": np.linspace(-0.10, 0.10, 41),
        "start_p": 7.0,
        "r0": 0.0,
        "eval_index": np.arange(rows),
    }
    result = train.run_directional_hmm(
        prepared,
        hmm,
        trigger,
        treatment,
        treatment_enabled=True,
    )
    expected_trigger, expected_active = (
        train.cusum_activation_schedule_from_innovation(
            result["innovation"],
            drift_allowance=trigger["drift_allowance_rate_cells"],
            positive_threshold=trigger["positive_threshold_rate_cells"],
            negative_threshold=trigger["negative_threshold_rate_cells"],
            tie_tolerance=trigger["tie_tolerance"],
            activation_transitions=trigger["activation_transitions"],
            refractory_rows=trigger["refractory_rows"],
        )
    )
    np.testing.assert_array_equal(result["trigger_direction"], expected_trigger)
    np.testing.assert_array_equal(result["active_direction"], expected_active)
    assert np.isfinite(result["posterior_mean"]).all()
    assert (
        result["maximum_normalization_error"]
        <= config["validation"]["stage_0"]["technical"][
            "normalization_max_abs_error"
        ]
    )


def test_future_direction_uses_fixed_past_and_future_physical_rate_windows(train):
    rows = 80
    rate = np.r_[np.full(40, 0.01), np.full(40, 0.03)]
    md = np.arange(1, rows + 1, dtype=float) * 10.0
    z = np.zeros(rows)
    tvt = np.cumsum(rate * 10.0) + 12_000.0
    frozen = train.FrozenWell(
        well="abc",
        role="persistent",
        fold=2,
        eval_id=np.asarray([str(index) for index in range(rows)]),
        row_idx=np.arange(rows),
        raw_gr_missing=np.zeros(rows, dtype=bool),
        parent_prediction=np.zeros(rows),
        treatment_prediction=np.zeros(rows),
        predictive_rate_mean=np.zeros(rows),
        filtered_rate_mean=np.zeros(rows),
        innovation=np.zeros(rows),
        positive_cusum=np.zeros(rows),
        negative_cusum=np.zeros(rows),
        trigger_direction=np.where(np.arange(rows) == 39, 1, 0).astype(np.int8),
        active_direction=np.zeros(rows, dtype=np.int8),
        last_known_tvt=12_000.0,
        last_known_md=0.0,
        last_known_z=0.0,
        schedule_sha256="a",
        prediction_sha256="b",
        maximum_normalization_error=0.0,
        log_likelihood=0.0,
        elapsed_seconds=1.0,
        prefix_rows=100,
    )
    truth = pd.DataFrame(
        {
            "row_idx": np.arange(rows),
            "id": [str(index) for index in range(rows)],
            "MD": md,
            "Z": z,
            "TVT": tvt,
            "TVT_input": np.nan,
        }
    )
    readout = train.trigger_future_direction_readout(frozen, truth)
    assert len(readout) == 1
    assert bool(readout.loc[0, "eligible_future_direction"])
    assert readout.loc[0, "future_true_rate_direction"] == 1
    assert bool(readout.loc[0, "direction_agreement"])


def test_truth_and_episodes_are_blocked_until_all_predictions_freeze(train):
    ledger = train.LeakageLedger(expected_wells=2)
    ledger.freeze("a")
    with pytest.raises(RuntimeError, match="before all fixed32"):
        ledger.record_truth_late(10)
    ledger.freeze("b")
    ledger.record_truth_late(10)
    ledger.record_episode_late(2)
    assert ledger.truth_rows_before_all_freeze == 10
    assert ledger.truth_rows_after_all_freeze == 10
    assert ledger.episode_rows_after_all_freeze == 2


def test_inference_is_fail_closed(inference, config):
    assert inference.validate_disabled_inference(config) == {
        "stage_1_execution_approved": False,
        "inference_enabled": False,
        "submission_enabled": False,
    }
    with pytest.raises(RuntimeError, match="inference is disabled"):
        inference.run_inference()


def test_notebook_source_is_self_contained_and_has_required_sections():
    source = TRAIN_SOURCE.read_text()
    assert "__file__" not in source
    assert "from settings import" not in source
    assert "from build_stage0_manifest import" not in source
    assert (
        source.count("well_metric_rows.append(well_truth_late_metrics(item, truth))")
        == 1
    )
    for heading in (
        "## 1. Imports and immutable execution contract",
        "## 3. Fixed32 manifest, saved parent, and target-free raw inputs",
        "## 5. Directional de-stick forward/backward kernel",
        "## 6. No-trigger parity and target-free prediction freeze",
        "## 7. Truth-late trigger and persistent-episode readout",
        "## 8. Stage 0 gates, generated artifacts, and metrics",
    ):
        assert heading in source
    parameters = inspect.signature(
        load_module(TRAIN_SOURCE, "exp411_train_signature").freeze_target_free_well
    ).parameters
    assert not {
        "truth",
        "error",
        "episodes",
        "fold",
        "role",
        "hidden_like_role",
    }.intersection(parameters)
