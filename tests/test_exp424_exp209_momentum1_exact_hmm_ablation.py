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
EXP_DIR = ROOT / "experiments" / "exp424_exp209_momentum1_exact_hmm_ablation"
TRAIN_SOURCE = EXP_DIR / "exp424_exp209_momentum1_exact_hmm_ablation_compact_selfcontained_train.py"
INFERENCE_SOURCE = (
    EXP_DIR / "exp424_exp209_momentum1_exact_hmm_ablation_compact_selfcontained_inference.py"
)
CONFIG_PATH = EXP_DIR / "config.yaml"
MANIFEST_PATH = (
    ROOT
    / "experiments"
    / "exp411_predictive_filtered_rate_innovation_destick"
    / "assets"
    / "stage0_fixed32_manifest.csv"
)
EPISODE_PATH = (
    ROOT
    / "experiments"
    / "exp408_hmm_message_rate_basin_audit"
    / "assets"
    / "persistent_offset_episodes.csv"
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
    return load_module(TRAIN_SOURCE, "exp424_train_test")


@pytest.fixture(scope="module")
def inference():
    return load_module(INFERENCE_SOURCE, "exp424_inference_test")


@pytest.fixture(scope="module")
def parent():
    return load_module(PARENT_SOURCE, "exp424_exp209_reference")


@pytest.fixture(scope="module")
def config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())


def test_completed_stage0_is_fail_closed_against_accidental_rerun(train, config):
    counts = train.validate_execution_contract(
        config,
        require_run_authorization=False,
    )
    assert counts == {
        "active_variants": 1,
        "stage_0_baseline_hmm_well_runs": 32,
        "stage_0_treatment_hmm_well_runs": 32,
        "stage_0_total_hmm_well_runs": 64,
        "parent_control_hmm_reruns_stage_0": 32,
        "planned_stage_1_treatment_hmm_well_runs": 773,
        "parent_control_hmm_reruns_stage_1": 0,
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
    assert config["design"]["kaggle_stage_0_completed"] is True
    assert config["design"]["kaggle_stage_0_all_gates_pass"] is False
    assert config["design"]["stage_1_eligible_for_separate_approval"] is False
    assert config["execution"]["run_hmm"] is False
    assert config["execution"]["create_prediction"] is False
    with pytest.raises(RuntimeError, match="execution.run_hmm is false"):
        train.validate_execution_contract(
            config,
            require_run_authorization=True,
        )

    unapproved = deepcopy(config)
    unapproved["design"]["kaggle_stage_0_authorized"] = False
    with pytest.raises(RuntimeError, match="does not authorize"):
        train.validate_execution_contract(
            unapproved,
            require_run_authorization=True,
        )


def test_contract_rejects_stage1_inference_submission_or_gpu(train, config):
    broken = deepcopy(config)
    broken["design"]["kaggle_stage_1_authorized"] = True
    with pytest.raises(ValueError, match="Stage 1"):
        train.validate_execution_contract(broken, require_run_authorization=False)

    broken = deepcopy(config)
    broken["design"]["inference_authorized"] = True
    with pytest.raises(ValueError, match="inference"):
        train.validate_execution_contract(broken, require_run_authorization=False)

    broken = deepcopy(config)
    broken["design"]["submission_authorized"] = True
    with pytest.raises(ValueError, match="submission"):
        train.validate_execution_contract(broken, require_run_authorization=False)

    broken = deepcopy(config)
    broken["runtime"]["enable_gpu"] = True
    with pytest.raises(ValueError, match="CPU-only"):
        train.validate_execution_contract(broken, require_run_authorization=False)


def test_scientific_contract_changes_only_momentum(train, config):
    contract = train.validate_scientific_contract(config)
    assert contract["changed_leaf_paths"] == ["mom"]
    assert contract["parent_hmm"]["mom"] == 0.998
    assert contract["treatment_hmm"]["mom"] == 1.0
    assert contract["parent_hmm"]["sig_r"] == 0.002
    assert contract["treatment_hmm"]["sig_r"] == 0.002
    assert {key: value for key, value in contract["parent_hmm"].items() if key != "mom"} == {
        key: value for key, value in contract["treatment_hmm"].items() if key != "mom"
    }

    broken = deepcopy(config)
    broken["model"]["treatment_hmm"]["sig_r"] = 0.004
    with pytest.raises(ValueError, match="treatment contract changed"):
        train.validate_scientific_contract(broken)


def test_fixed_inputs_are_sha_pinned_and_mechanism_only(config):
    manifest_sha = hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest()
    episode_sha = hashlib.sha256(EPISODE_PATH.read_bytes()).hexdigest()
    assert manifest_sha == config["data"]["stage_0_manifest"]["expected_sha256"]
    assert episode_sha == config["data"]["persistent_episodes"]["expected_sha256"]
    manifest = pd.read_csv(MANIFEST_PATH, dtype={"well": str})
    assert len(manifest) == 32
    assert manifest["well"].nunique() == 32
    assert manifest["role"].value_counts().to_dict() == {
        "control": 16,
        "persistent": 16,
    }
    assert manifest.groupby("fold").size().to_dict() == {
        0: 8,
        1: 6,
        2: 6,
        3: 6,
        4: 6,
    }
    assert config["design"]["stage_0_role"] == ("mechanism_preflight_only_not_cv_or_promotion")


def test_momentum1_removes_only_interior_zero_directed_rate_drift(train, config):
    parent_hmm = config["model"]["parent_hmm"]
    treatment_hmm = config["model"]["treatment_hmm"]
    rates = np.linspace(-0.10, 0.10, 41)
    source_index = 28
    source_rate = rates[source_index]
    parent_expected = train.rate_kernel_expected_destination(
        rates,
        source_index=source_index,
        dm=1.0,
        sig_r=parent_hmm["sig_r"],
        mom=parent_hmm["mom"],
    )
    treatment_expected = train.rate_kernel_expected_destination(
        rates,
        source_index=source_index,
        dm=1.0,
        sig_r=treatment_hmm["sig_r"],
        mom=treatment_hmm["mom"],
    )
    assert parent_expected == pytest.approx(0.998 * source_rate, abs=1e-12)
    assert treatment_expected == pytest.approx(source_rate, abs=1e-12)

    parent_kernel = train.rate_kernel_probabilities(
        rates,
        1.0,
        parent_hmm["sig_r"],
        parent_hmm["mom"],
    )
    treatment_kernel = train.rate_kernel_probabilities(
        rates,
        1.0,
        treatment_hmm["sig_r"],
        treatment_hmm["mom"],
    )
    np.testing.assert_allclose(parent_kernel.sum(axis=1), 1.0, atol=1e-14)
    np.testing.assert_allclose(treatment_kernel.sum(axis=1), 1.0, atol=1e-14)
    assert treatment_kernel[source_index, 0] == pytest.approx(treatment_kernel[source_index, 2])


def test_parent_variant_matches_independent_exp209_small_trellis(
    train,
    parent,
    config,
):
    rng = np.random.default_rng(424)
    row_count = 20
    positions = 13
    emission = rng.normal(0.0, 0.3, size=(row_count, positions)).astype(np.float32)
    dm = 1.0 + (np.arange(row_count, dtype=np.float64) % 5) * 0.2
    dz = 0.25 * np.sin(np.arange(row_count, dtype=np.float64) / 7.0)
    rates = np.linspace(-0.06, 0.06, 9, dtype=np.float64)
    prepared = {
        "emission_ll": emission,
        "dm": dm,
        "dz": dz,
        "grid": 11_900.0 + np.arange(positions) * 0.35,
        "rates": rates,
        "start_p": 6.0,
        "r0": 0.0,
        "eval_index": np.arange(row_count),
    }
    observed = train.run_hmm_variant(
        prepared,
        config["model"]["parent_hmm"],
    )
    reference_position, reference_loglik = parent._hmm2_fb(
        emission,
        dm,
        dz,
        0.35,
        rates,
        0.002,
        0.02,
        6.0,
        0.75,
        0.0,
        0.01,
        1.0,
        0.998,
    )
    reference_mean = reference_position @ prepared["grid"]
    np.testing.assert_allclose(
        observed["posterior_mean"],
        reference_mean,
        rtol=0.0,
        atol=2e-7,
    )
    assert abs(observed["log_likelihood"] - reference_loglik) <= 2e-6
    assert observed["maximum_normalization_error"] <= 1e-6
    assert np.isfinite(observed["smoothed_rate_mean"]).all()
    assert np.all(observed["smoothed_rate_edge_mass"] >= 0.0)
    assert np.all(observed["smoothed_rate_edge_mass"] <= 1.0)


def test_underresponse_definition_is_fixed_and_sse_weighted(train):
    true_rate = np.asarray([0.04, -0.04, 0.02, np.nan])
    estimated = np.asarray([0.02, -0.05, -0.01, 0.0])
    result = train.zero_direction_underresponse_stats(true_rate, estimated)
    assert result["valid_rows"] == 3
    assert result["underresponse_rows"] == 1
    assert result["rate_error_sse"] == pytest.approx(
        (0.02 - 0.04) ** 2 + (-0.05 + 0.04) ** 2 + (-0.01 - 0.02) ** 2
    )
    assert result["underresponse_sse"] == pytest.approx((0.02 - 0.04) ** 2)


def test_truth_and_scope_fields_cannot_enter_decoder(train):
    source = inspect.getsource(train.load_target_free_well)
    assert 'str(column) != "TVT"' in source
    assert "FORBIDDEN_DECODER_COLUMNS.intersection" in source
    assert {"TVT", "error", "episode_id", "fold", "role"}.issubset(train.FORBIDDEN_DECODER_COLUMNS)
    ledger = train.LeakageLedger(expected_wells=2)
    ledger.freeze("a")
    with pytest.raises(RuntimeError, match="before all"):
        ledger.record_truth_late(3)
    assert ledger.truth_rows_before_all_freeze == 3


def test_inference_is_fail_closed(inference, config):
    contract = inference.validate_inference_disabled(config)
    assert contract["stage0_completed"] is True
    assert contract["stage0_all_gates_pass"] is False
    assert contract["inference_authorized"] is False
    assert contract["submission_authorized"] is False
    with pytest.raises(RuntimeError, match="inference is disabled"):
        inference.run_inference(config)


def test_notebook_sources_are_self_contained_and_notebook_safe():
    train_source = TRAIN_SOURCE.read_text()
    inference_source = INFERENCE_SOURCE.read_text()
    assert "Path(__file__)" not in train_source
    assert "__file__" not in train_source
    assert "Path(__file__)" not in inference_source
    assert "__file__" not in inference_source
    assert "from settings import" not in train_source
    assert "from settings import" not in inference_source
    assert "exp411_directional_trigger" not in train_source
    assert "beta_filter_activation_schedule" not in train_source
    assert "# ## 5. Exact forward-backward kernel" in train_source
    assert "# ## 8. Stage 0 gates" in train_source
