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
    numba_stub.__version__ = "test-stub"
    sys.modules["numba"] = numba_stub

ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "experiments" / "exp425_symmetric_datum_reanchor_exact_hmm"
TRAIN_SOURCE = (
    EXP_DIR
    / "exp425_symmetric_datum_reanchor_exact_hmm_compact_selfcontained_train.py"
)
INFERENCE_SOURCE = (
    EXP_DIR
    / "exp425_symmetric_datum_reanchor_exact_hmm_compact_selfcontained_inference.py"
)
CONFIG_PATH = EXP_DIR / "config.yaml"
MANIFEST_PATH = (
    ROOT
    / "experiments"
    / "exp412_beta_filter_rate_disagreement_two_pass_reset"
    / "assets"
    / "stage0_fixed32_manifest.csv"
)
MANIFEST_METADATA_PATH = MANIFEST_PATH.with_name(
    "stage0_fixed32_manifest_metadata.json"
)
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
    return load_module(TRAIN_SOURCE, "exp425_train_test")


@pytest.fixture(scope="module")
def inference():
    return load_module(INFERENCE_SOURCE, "exp425_inference_test")


@pytest.fixture(scope="module")
def parent():
    return load_module(PARENT_SOURCE, "exp425_exp209_reference")


@pytest.fixture(scope="module")
def config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())


def synthetic_prepared(rows: int = 18, positions: int = 15) -> dict:
    x = np.linspace(-1.0, 1.0, positions)
    emission = np.vstack(
        [
            -0.5 * ((x - 0.45 * np.sin(row / 3.0)) / 0.38) ** 2
            for row in range(rows)
        ]
    ).astype(np.float32)
    return {
        "emission_ll": emission,
        "dm": 8.0 + (np.arange(rows, dtype=np.float64) % 4) * 2.0,
        "dz": 0.2 * np.cos(np.arange(rows, dtype=np.float64) / 4.0),
        "grid": 11_900.0 + np.arange(positions, dtype=np.float64) * 0.35,
        "rates": np.linspace(-0.06, 0.06, 9, dtype=np.float64),
        "start_p": 7.0,
        "r0": 0.0,
        "eval_index": np.arange(rows, dtype=np.int64),
    }


def test_completed_stage0_is_fail_closed_while_later_actions_remain_locked(
    train,
    config,
):
    counts = train.validate_execution_contract(
        config,
        require_run_authorization=False,
    )
    assert counts == {
        "active_scientific_variants": 1,
        "planned_stage_0_baseline_hmm_well_runs": 32,
        "planned_stage_0_treatment_hmm_logical_well_runs": 32,
        "planned_stage_0_total_logical_hmm_well_runs": 64,
        "planned_stage_0_treatment_branch_states": 3,
        "parent_control_hmm_reruns_stage_0": 32,
        "lightgbm_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
        "models": 0,
        "pf_runs": 0,
        "beam_runs": 0,
        "gpu_runs": 0,
    }
    assert config["design"]["implementation_authorized"] is True
    assert config["design"]["canonical_notebook_adoption_authorized"] is True
    assert config["design"]["kaggle_stage_0_authorized"] is True
    assert (
        config["design"]["kaggle_stage_0_approval_source"]
        == "user_message_execute_exp425_2026_07_28"
    )
    assert config["design"]["kaggle_stage_0_completed"] is True
    assert config["design"]["kaggle_stage_0_result"] == "stage0_fail_closed"
    assert config["design"]["stage_1_eligible"] is False
    assert config["execution"]["run_hmm"] is False
    assert config["design"]["kaggle_stage_1_authorized"] is False
    assert config["design"]["inference_authorized"] is False
    assert config["design"]["submission_authorized"] is False
    with pytest.raises(RuntimeError, match="execution.run_hmm is false"):
        train.validate_execution_contract(
            config,
            require_run_authorization=True,
        )


def test_contract_rejects_stage1_inference_submission_or_gpu(train, config):
    for key in (
        "kaggle_stage_1_authorized",
        "inference_authorized",
        "submission_authorized",
    ):
        broken = deepcopy(config)
        broken["design"][key] = True
        with pytest.raises(ValueError):
            train.validate_execution_contract(
                broken,
                require_run_authorization=False,
            )
    broken = deepcopy(config)
    broken["runtime"]["enable_gpu"] = True
    with pytest.raises(ValueError, match="CPU-only"):
        train.validate_execution_contract(
            broken,
            require_run_authorization=False,
        )


def test_scientific_contract_pins_symmetric_single_event_branch(train, config):
    contract = train.validate_scientific_contract(config)
    assert contract["parent_hmm"]["mom"] == 0.998
    assert contract["parent_hmm"]["sig_r"] == 0.002
    assert contract["trigger"]["maximum_events_per_well"] == 1
    assert contract["trigger"]["rate_sign_selects_datum_direction"] is False
    assert contract["datum_branch"]["explicit_state_values"] == [
        "negative",
        "parent",
        "positive",
    ]
    assert contract["datum_branch"]["prior_mass"] == {
        "negative": 0.10,
        "parent": 0.80,
        "positive": 0.10,
    }
    assert contract["datum_branch"]["allow_branch_switch_after_event"] is False

    broken = deepcopy(config)
    broken["model"]["datum_branch"]["prior_mass"]["positive"] = 0.11
    with pytest.raises(ValueError, match="branch contract changed"):
        train.validate_scientific_contract(broken)


def test_fixed32_manifest_is_sha_pinned_and_mechanism_only(train, config):
    observed = hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest()
    assert observed == config["data"]["stage_0_manifest"]["expected_sha256"]
    metadata_sha = hashlib.sha256(MANIFEST_METADATA_PATH.read_bytes()).hexdigest()
    assert (
        metadata_sha
        == config["data"]["stage_0_manifest_metadata"]["expected_sha256"]
    )
    manifest = pd.read_csv(MANIFEST_PATH, dtype={"well": str})
    assert len(manifest) == 32
    assert manifest["well"].nunique() == 32
    assert manifest["role"].value_counts().to_dict() == {
        "control": 16,
        "backward_cause": 8,
        "forward_cause": 8,
    }
    assert (
        config["design"]["stage_0_role"]
        == "mechanism_preflight_only_not_cv_or_promotion"
    )
    loaded, evidence = train.load_fixed32_manifest(
        config,
        train.LeakageLedger(expected_wells=32),
    )
    assert len(loaded) == 32
    assert evidence["metadata_sha256"] == metadata_sha


def test_zero_position_shift_matches_independent_exp209(
    train,
    parent,
    config,
):
    prepared = synthetic_prepared()
    observed = train.run_hmm_pass(
        prepared,
        config["model"]["parent_hmm"],
        position_shift_ft=np.zeros(len(prepared["eval_index"])),
    )
    reference_position, reference_loglik = parent._hmm2_fb(
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
    reference_mean = reference_position @ prepared["grid"]
    np.testing.assert_allclose(
        observed["posterior_mean"],
        reference_mean,
        rtol=0.0,
        atol=2.0e-7,
    )
    assert abs(observed["log_likelihood"] - reference_loglik) <= 2.0e-6
    assert np.isfinite(observed["filtered_position_std"]).all()
    assert np.all(observed["filtered_position_std"] >= 0.0)


def test_first_persistent_event_uses_activation_only(train, config):
    trigger = config["model"]["trigger"]
    filtered = np.zeros(24)
    filtered_std = np.zeros(24)
    positive = train.beta_filter_activation_schedule(
        filtered,
        filtered_std,
        np.full(24, 0.01),
        trigger,
    )
    negative = train.beta_filter_activation_schedule(
        filtered,
        filtered_std,
        np.full(24, -0.01),
        trigger,
    )
    assert train.first_persistent_activation_event(positive["active_direction"]) == 7
    assert train.first_persistent_activation_event(negative["active_direction"]) == 7
    assert positive["active_direction"][7] == 1
    assert negative["active_direction"][7] == -1
    assert train.first_persistent_activation_event(np.zeros(24, dtype=np.int8)) == -1


def test_no_event_is_exact_parent_only_branch(train, config):
    prepared = synthetic_prepared()
    baseline = train.run_hmm_pass(
        prepared,
        config["model"]["parent_hmm"],
        position_shift_ft=np.zeros(len(prepared["eval_index"])),
    )
    observed = train.run_symmetric_datum_treatment(
        prepared,
        config["model"]["parent_hmm"],
        config["model"]["datum_branch"],
        baseline=baseline,
        event_index=-1,
        datum_shift_ft=0.35,
    )
    np.testing.assert_array_equal(
        observed["posterior_mean"],
        baseline["posterior_mean"],
    )
    np.testing.assert_array_equal(
        observed["branch_posterior_final"],
        np.asarray([0.0, 1.0, 0.0]),
    )
    assert observed["prediction_sha256"] == baseline["prediction_sha256"]


def test_symmetric_branch_marginal_is_exact_soft_evidence_mixture(train, config):
    prepared = synthetic_prepared(rows=16, positions=17)
    hmm = config["model"]["parent_hmm"]
    branch = config["model"]["datum_branch"]
    baseline = train.run_hmm_pass(
        prepared,
        hmm,
        position_shift_ft=np.zeros(len(prepared["eval_index"])),
    )
    event = 6
    shift = 0.70
    observed = train.run_symmetric_datum_treatment(
        prepared,
        hmm,
        branch,
        baseline=baseline,
        event_index=event,
        datum_shift_ft=shift,
    )
    weights = observed["branch_posterior_final"]
    np.testing.assert_allclose(weights.sum(), 1.0, atol=1.0e-12)
    assert np.all(weights > 0.0)
    np.testing.assert_array_equal(
        observed["branch_posterior_mass"][:event],
        np.tile(np.asarray([0.0, 1.0, 0.0]), (event, 1)),
    )
    np.testing.assert_allclose(
        observed["branch_posterior_mass"][event:],
        np.tile(weights, (len(prepared["eval_index"]) - event, 1)),
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        observed["posterior_mean"],
        weights @ observed["conditional_prediction"],
        atol=1.0e-12,
    )
    prior = np.asarray([0.10, 0.80, 0.10])
    manual_weights, manual_evidence = train.normalized_log_weights(
        np.log(prior) + observed["conditional_log_likelihood"]
    )
    np.testing.assert_allclose(weights, manual_weights, atol=1.0e-12)
    assert observed["log_likelihood"] == pytest.approx(manual_evidence)
    assert observed["maximum_normalization_error"] <= 1.0e-5


def test_truth_and_episode_reads_are_blocked_until_every_well_freezes(train):
    ledger = train.LeakageLedger(expected_wells=2)
    ledger.freeze("a")
    with pytest.raises(RuntimeError, match="before all fixed32"):
        ledger.record_truth_late(3)
    with pytest.raises(RuntimeError, match="before all fixed32"):
        ledger.record_episode_late(1)
    ledger.freeze("b")
    ledger.record_truth_late(3)
    ledger.record_episode_late(1)
    assert ledger.truth_rows_before_all_freeze == 3
    assert ledger.episode_rows_before_all_freeze == 1


def test_stage0_gate_is_strict_and_uses_branch_mass_contract(train, config):
    roles = ["backward_cause"] * 8 + ["forward_cause"] * 8 + ["control"] * 16
    folds = [0, 1, 2, 3, 4, 0, 1, 2] * 2 + [0, 1, 2, 3, 4] * 3 + [0]
    manifest = pd.DataFrame(
        {
            "well": [f"well_{index:02d}" for index in range(32)],
            "role": roles,
            "fold": folds,
        }
    )
    branch_mass = np.tile(np.asarray([0.05, 0.90, 0.05]), (4, 1))
    frozen = [
        types.SimpleNamespace(
            role=role,
            fold=fold,
            row_idx=np.arange(4),
            event_index=1,
            baseline_saved_parent_max_abs_diff_ft=0.0,
            maximum_normalization_error=1.0e-8,
            baseline_prediction=np.ones(4),
            treatment_prediction=np.ones(4),
            filtered_position_std=np.ones(4),
            branch_posterior_mass=branch_mass,
        )
        for role, fold in zip(roles, folds, strict=True)
    ]
    direction = pd.DataFrame(
        {
            "fold": np.repeat(np.arange(5), 2),
            "eligible_soft_datum_direction": True,
            "direction_agreement": True,
        }
    )
    cause = pd.DataFrame(
        [
            {
                "role": "backward_cause",
                "rows": 10,
                "baseline_sse": 100.0,
                "treatment_sse": 80.0,
                "post_event_rows": 8,
            },
            {
                "role": "forward_cause",
                "rows": 10,
                "baseline_sse": 100.0,
                "treatment_sse": 101.0,
                "post_event_rows": 8,
            },
        ]
    )
    well_metrics = pd.DataFrame(
        {
            "role": ["control"] * 16,
            "rows": [4] * 16,
            "baseline_rmse_ft": [1.0] * 16,
            "treatment_rmse_ft": [1.0] * 16,
        }
    )
    ledger = train.LeakageLedger(expected_wells=32)
    for well in manifest["well"]:
        ledger.freeze(well)
    gates = train.evaluate_stage0_gates(
        config=config,
        manifest=manifest,
        frozen_wells=frozen,
        parity={"pass": True},
        schedule_artifact={
            "logical_sha256": "same",
            "readback_logical_sha256": "same",
        },
        direction_readout=direction,
        cause_readout=cause,
        well_metrics=well_metrics,
        ledger=ledger,
        elapsed_seconds=1.0,
    )
    assert all(gates["technical"].values())
    assert all(gates["mechanism"].values())
    assert gates["stage_1_eligible"] is True

    unsafe = well_metrics.copy()
    unsafe["treatment_rmse_ft"] = 1.03
    failed = train.evaluate_stage0_gates(
        config=config,
        manifest=manifest,
        frozen_wells=frozen,
        parity={"pass": True},
        schedule_artifact={
            "logical_sha256": "same",
            "readback_logical_sha256": "same",
        },
        direction_readout=direction,
        cause_readout=cause,
        well_metrics=unsafe,
        ledger=ledger,
        elapsed_seconds=1.0,
    )
    assert failed["mechanism"]["matched_control_rmse_safety"] is False
    assert failed["stage_1_eligible"] is False


def test_inference_is_fail_closed(inference, config):
    assert inference.validate_disabled_inference(config) == {
        "stage_1_execution_approved": False,
        "inference_enabled": False,
        "submission_enabled": False,
    }
    with pytest.raises(RuntimeError, match="inference is disabled"):
        inference.run_inference()


def test_notebook_source_is_self_contained_and_truth_late():
    source = TRAIN_SOURCE.read_text()
    inference_source = INFERENCE_SOURCE.read_text()
    assert "__file__" not in source
    assert "__file__" not in inference_source
    assert "from settings import" not in source
    assert "from exact_hmm_smoother import" not in source
    for heading in (
        "## 1. Imports and immutable execution contract",
        "## 3. Fixed32 manifest, saved parent, and target-free raw inputs",
        "## 5. Exact position-shift HMM and symmetric branch marginalization",
        "## 6. Parent parity, event freeze, and target-free prediction freeze",
        "## 7. Truth-late datum-direction, cause, and safety readout",
        "## 8. Stage 0 gates, generated artifacts, and metrics",
    ):
        assert heading in source
    parameters = inspect.signature(
        load_module(TRAIN_SOURCE, "exp425_train_signature").freeze_target_free_well
    ).parameters
    assert not {
        "truth",
        "error",
        "episodes",
        "cause",
        "fold",
        "role",
        "hidden_like_role",
    }.intersection(parameters)
