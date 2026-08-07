from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.util
import inspect
import os
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
EXP = "exp432_symmetric_datum_defensive_particle_reinjection"
EXP_DIR = ROOT / "experiments" / EXP
SOURCE = EXP_DIR / f"{EXP}_compact_selfcontained_train.py"
CONFIG_PATH = EXP_DIR / "config.yaml"
MANIFEST_PATH = (
    ROOT
    / "experiments"
    / "exp412_beta_filter_rate_disagreement_two_pass_reset"
    / "assets"
    / "stage0_fixed32_manifest.csv"
)
EXP404_SOURCE = (
    ROOT
    / "experiments"
    / "exp404_scale5_sigma_gr_likelihood_pf_ablation"
    / "exp404_scale5_sigma_gr_likelihood_pf_ablation_compact_selfcontained_train.py"
)
EXP209_SOURCE = (
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
    previous = os.environ.get("EXP432_IMPORT_ONLY")
    os.environ["EXP432_IMPORT_ONLY"] = "1"
    try:
        return load_module(SOURCE, "exp432_train_test")
    finally:
        if previous is None:
            os.environ.pop("EXP432_IMPORT_ONLY", None)
        else:
            os.environ["EXP432_IMPORT_ONLY"] = previous


@pytest.fixture(scope="module")
def exp404():
    return load_module(EXP404_SOURCE, "exp404_parent_for_exp432")


@pytest.fixture(scope="module")
def exp209():
    return load_module(EXP209_SOURCE, "exp209_parent_for_exp432")


@pytest.fixture(scope="module")
def config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())


def synthetic_pf_inputs() -> tuple[np.ndarray, ...]:
    md = np.arange(1.0, 9.0, dtype=np.float64)
    z = np.linspace(0.0, 0.7, len(md), dtype=np.float64)
    gr = np.asarray([50.0, 52.0, 54.0, 53.0, 51.0, 50.0, 49.0, 51.0])
    grid_gr = np.linspace(40.0, 70.0, 151, dtype=np.float64)
    return md, z, gr, grid_gr


def run_exp432_kernel(
    train,
    *,
    event_index: int,
    datum: float,
    components: np.ndarray,
) -> tuple[np.ndarray, ...]:
    md, z, gr, grid_gr = synthetic_pf_inputs()
    return train._pf_symmetric_allseeds(
        md,
        z,
        gr,
        grid_gr,
        90.0,
        0.2,
        20.0,
        100.0,
        0.01,
        components.shape[1],
        components.shape[0],
        12345,
        0.998,
        0.002,
        0.005,
        0.1,
        0.001,
        0.5,
        4.5,
        0.01,
        event_index,
        datum,
        components,
    )


def test_stage0_completed_fail_closed_and_rerun_remains_locked(
    train, config
):
    counts = train.validate_execution_contract(
        config, require_run_authorization=False
    )
    assert counts["active_scientific_variants"] == 1
    assert counts["stage_0_hmm_trigger_well_runs"] == 32
    assert counts["stage_0_baseline_pf_well_runs"] == 32
    assert counts["stage_0_treatment_pf_well_runs"] == 32
    assert counts["stage_0_total_pf_well_runs"] == 64
    assert counts["stage_0_seed_well_trajectories"] == 8192
    assert counts["stage_0_particle_starts"] == 4096000
    assert counts["lightgbm_configs"] == 0
    assert counts["boosters"] == 0
    assert counts["gpu_runs"] == 0
    assert config["experiment"]["status"] == "stage0_fail_closed"
    assert config["design"]["implementation_allowed_now"] is True
    assert config["design"]["canonical_notebook_adoption_allowed"] is True
    assert config["design"]["kaggle_stage0_push_allowed"] is False
    assert config["design"]["full_execution_allowed"] is False
    assert config["design"]["inference_allowed"] is False
    assert config["design"]["submission_allowed"] is False
    assert config["execution"]["selected_stage"] is None
    assert config["execution"]["kaggle_execution_authorized"] is False
    assert config["execution"]["stage0_completed"] is True
    assert config["execution"]["stage0_mechanism_pass"] is False
    assert config["execution"]["full_eligible"] is False
    with pytest.raises(RuntimeError, match="not approved"):
        train.validate_execution_contract(config, require_run_authorization=True)


def test_scientific_contract_pins_direction_free_single_event_proposal(train, config):
    contract = train.validate_scientific_contract(config)
    assert contract["proposal_mass"] == [0.8, 0.1, 0.1]
    assert contract["trigger"]["maximum_events_per_well"] == 1
    assert contract["trigger"]["beta_sign_used_by_treatment"] is False
    assert contract["importance"]["clip"] is None
    assert (
        contract["importance"]["weight_update"]
        == "add_finite_log_p0_minus_log_q_then_log_normalize"
    )
    assert contract["pf"]["particles"] == 500
    assert contract["pf"]["seeds"] == 128
    assert contract["pf"]["temperature"] == 5.0
    assert len(contract["sha256"]) == 64

    broken = deepcopy(config)
    broken["proposal"]["components"]["plus_datum"]["mass"] = 0.11
    with pytest.raises(ValueError, match="mass contract"):
        train.validate_scientific_contract(broken)


def test_fixed32_manifest_is_sha_pinned_unique_and_balanced(config):
    observed = hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest()
    assert observed == config["data"]["stage_0_manifest"]["expected_sha256"]
    manifest = pd.read_csv(MANIFEST_PATH, dtype={"well": str})
    assert len(manifest) == 32
    assert manifest["well"].nunique() == 32
    assert manifest["role"].value_counts().to_dict() == {
        "control": 16,
        "backward_cause": 8,
        "forward_cause": 8,
    }
    assert set(manifest["fold"].astype(int)) == set(range(5))


def test_beta_schedule_uses_inclusive_window_and_first_false_to_true_event(train):
    trigger = {
        "denominator_floor": 0.005,
        "absolute_z_threshold": 2.0,
        "rolling_window_rows": 16,
        "qualifying_rows_min": 8,
        "same_sign_fraction_min": 0.75,
    }
    filtered = np.zeros(24)
    std = np.full(24, 0.005)
    smoothed = np.zeros(24)
    smoothed[3:11] = 0.015
    schedule = train.beta_filter_activation_schedule(
        smoothed, filtered, std, trigger
    )
    assert schedule["active_direction"][9] == 0
    assert schedule["active_direction"][10] == 1
    assert train.first_persistent_activation_event(
        schedule["active_direction"]
    ) == 10
    mixed = np.asarray([1, -1] * 8, dtype=np.float64) * 0.015
    tied = train.beta_filter_activation_schedule(
        mixed, np.zeros(16), np.full(16, 0.005), trigger
    )
    assert (tied["active_direction"] == 0).all()


def test_first_pass_hmm_matches_independent_exp209_kernel(train, exp209, config):
    rows = 18
    positions = 15
    x = np.linspace(-1.0, 1.0, positions)
    emission = np.vstack(
        [
            -0.5 * ((x - 0.45 * np.sin(row / 3.0)) / 0.38) ** 2
            for row in range(rows)
        ]
    ).astype(np.float32)
    prepared = {
        "emission_ll": emission,
        "dm": 8.0 + (np.arange(rows, dtype=np.float64) % 4) * 2.0,
        "dz": 0.2 * np.cos(np.arange(rows, dtype=np.float64) / 4.0),
        "grid": 11_900.0 + np.arange(positions, dtype=np.float64) * 0.35,
        "rates": np.linspace(-0.06, 0.06, 9, dtype=np.float64),
        "start_p": 7.0,
        "r0": 0.0,
        "eval_indices": np.arange(rows, dtype=np.int64),
    }
    observed = train.run_first_pass_hmm(prepared, config["hmm"])
    reference_position, reference_loglik = exp209._hmm2_fb(
        prepared["emission_ll"],
        prepared["dm"],
        prepared["dz"],
        0.35,
        prepared["rates"],
        0.002,
        0.02,
        prepared["start_p"],
        0.75,
        prepared["r0"],
        0.01,
        1.0,
        0.998,
    )
    np.testing.assert_allclose(
        observed["prediction"],
        reference_position @ prepared["grid"],
        rtol=0.0,
        atol=1.0e-10,
    )
    assert observed["log_likelihood"] == pytest.approx(
        reference_loglik, abs=1.0e-10
    )


def test_component_rng_is_stable_symmetric_and_particle_keyed(train):
    first = train.component_draws(
        "well-a", seeds=64, particles=500, event_row=17
    )
    second = train.component_draws(
        "well-a", seeds=64, particles=500, event_row=17
    )
    changed = train.component_draws(
        "well-a", seeds=64, particles=500, event_row=18
    )
    assert np.array_equal(first, second)
    assert not np.array_equal(first, changed)
    fractions = np.bincount(first.ravel(), minlength=3) / first.size
    np.testing.assert_allclose(fractions, [0.8, 0.1, 0.1], atol=0.01)


def test_log_importance_is_finite_unclipped_and_bounded(train):
    values = []
    for center in (0.0, -0.35, 0.35):
        for offset in np.linspace(-0.03, 0.03, 101):
            values.append(
                train.symmetric_position_log_importance(
                    float(center + offset), 0.35, 0.005
                )
            )
    log_importance = np.asarray([value[0] for value in values])
    log_q = np.asarray([value[1] for value in values])
    assert np.isfinite(log_importance).all()
    assert np.isfinite(log_q).all()
    assert float(log_importance.max()) <= np.log(1.25) + 1.0e-12
    assert float(log_importance.min()) < np.log(np.finfo(np.float64).tiny)


def test_pf_input_preparation_matches_exp404_with_missing_gr(
    train, exp404, config
):
    horizontal = pd.DataFrame(
        {
            "MD": np.arange(12, dtype=np.float64) * 10.0,
            "Z": np.linspace(0.0, 1.1, 12),
            "GR": [50.0, np.nan, 52.0, 54.0, 51.0, 49.0, 48.0, np.nan, 53.0, 55.0, 54.0, 52.0],
            "TVT_input": [100.0, 100.5, 101.0, 101.5, 102.0, 102.5, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        }
    )
    typewell = pd.DataFrame(
        {
            "TVT": np.linspace(95.0, 110.0, 31),
            "GR": np.r_[np.linspace(45.0, 60.0, 15), np.nan, np.linspace(61.0, 75.0, 15)],
        }
    )
    expected = exp404.prepare_likelihood_pf_inputs(
        horizontal,
        typewell.assign(GR=typewell["GR"].fillna(typewell["GR"].mean())),
        multiplier=1.0,
        grid_step=0.2,
    )
    observed = train.prepare_pf_inputs(
        horizontal,
        typewell,
        config["pf"]["dynamics"],
    )
    for key in (
        "eval_indices",
        "eval_md",
        "eval_z",
        "eval_gr",
        "grid_gr",
    ):
        np.testing.assert_array_equal(observed[key], expected[key])
    for key in (
        "last_known_position",
        "initial_rate",
        "grid_minimum",
        "grid_step",
    ):
        assert observed[key] == expected[key]
    assert observed["gr_scale"] == expected["scale_audit"]["candidate_scale"]


def test_importance_quadrature_recovers_parent_moments(train):
    report = train.importance_quadrature_contract(points=80_001)
    assert report["pass"] is True
    assert report["mass"] == pytest.approx(1.0, abs=1.0e-8)
    assert report["mean"] == pytest.approx(0.0, abs=1.0e-10)
    assert report["variance"] == pytest.approx(0.005**2, abs=1.0e-9)


def test_no_event_kernel_is_exact_exp404_rng_and_prediction_parity(train, exp404):
    md, z, gr, grid_gr = synthetic_pf_inputs()
    expected = exp404._pf_lik_allseeds(
        md,
        z,
        gr,
        grid_gr,
        90.0,
        0.2,
        20.0,
        100.0,
        0.01,
        24,
        4,
        12345,
        0.998,
        0.002,
        0.005,
        0.1,
        0.001,
        0.5,
        4.5,
    )
    observed = run_exp432_kernel(
        train,
        event_index=-1,
        datum=0.0,
        components=np.zeros((4, 24), dtype=np.int8),
    )
    assert np.array_equal(observed[0], expected[0])
    assert np.array_equal(observed[1], expected[1])
    assert np.array_equal(observed[9], expected[2])
    assert np.array_equal(observed[10], expected[3])
    assert np.array_equal(observed[11], expected[4])


def test_event_changes_only_entering_transition_and_preserves_pre_event_rng(train):
    components = train.component_draws(
        "synthetic-event", seeds=4, particles=24, event_row=4
    )
    baseline = run_exp432_kernel(
        train,
        event_index=-1,
        datum=0.0,
        components=components,
    )
    treatment = run_exp432_kernel(
        train,
        event_index=4,
        datum=0.35,
        components=components,
    )
    assert np.array_equal(baseline[0][:, :4], treatment[0][:, :4])
    assert np.array_equal(baseline[2][:, :4], treatment[2][:, :4])
    assert np.array_equal(baseline[3][:, :4], treatment[3][:, :4])
    assert np.isfinite(treatment[6]).all()
    assert np.isfinite(treatment[7]).all()
    assert float(treatment[7].max()) <= np.log(1.25) + 1.0e-12
    assert treatment[5].sum() == 4 * 24
    assert treatment[4][4, 1] > 0.0
    assert treatment[4][4, 2] > 0.0


def test_source_is_not_a_thin_helper_notebook_and_has_truth_late_guard(train):
    source = SOURCE.read_text()
    headings = [
        "Exact exp209 first-pass HMM",
        "Exp404 likelihood-PF input preparation",
        "Symmetric defensive proposal",
        "Target-free Stage 0 prediction freeze",
        "Truth-late mechanism readout",
        "Generated artifacts",
    ]
    for heading in headings:
        assert heading in source
    assert "Path(__file__)" not in source
    assert "from settings import" not in source
    assert "truth was read before all fixed32 artifacts froze" in source
    assert "cause roles/folds were read before all artifacts froze" in source
    assert "beta_sign_used_by_treatment" in inspect.getsource(
        train.validate_scientific_contract
    )
