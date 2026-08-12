from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
EXP = "exp459_persistent_acceleration_state_likelihood_pf"
EXP_DIR = ROOT / "experiments" / EXP
SOURCE = EXP_DIR / f"{EXP}_compact_selfcontained_train.py"
CONFIG_PATH = EXP_DIR / "config.yaml"
CANONICAL_TRAIN = EXP_DIR / f"{EXP}_train.ipynb"
COMPACT_TRAIN = EXP_DIR / f"{EXP}_compact_selfcontained_train.ipynb"
MANIFEST_PATH = (
    ROOT
    / "experiments"
    / "exp411_predictive_filtered_rate_innovation_destick"
    / "assets"
    / "stage0_fixed32_manifest.csv"
)
EXP404_SOURCE = (
    ROOT
    / "experiments"
    / "exp404_scale5_sigma_gr_likelihood_pf_ablation"
    / "exp404_scale5_sigma_gr_likelihood_pf_ablation_compact_selfcontained_train.py"
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
    previous = os.environ.get("EXP459_IMPORT_ONLY")
    os.environ["EXP459_IMPORT_ONLY"] = "1"
    try:
        return load_module(SOURCE, "exp459_train_test")
    finally:
        if previous is None:
            os.environ.pop("EXP459_IMPORT_ONLY", None)
        else:
            os.environ["EXP459_IMPORT_ONLY"] = previous


@pytest.fixture(scope="module")
def exp404():
    return load_module(EXP404_SOURCE, "exp404_parent_for_exp459")


@pytest.fixture(scope="module")
def config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())


def synthetic_pf_inputs() -> tuple[np.ndarray, ...]:
    md = np.arange(1.0, 10.0, dtype=np.float64)
    z = np.linspace(0.0, 0.8, len(md), dtype=np.float64)
    gr = np.asarray(
        [50.0, 52.0, 54.0, 53.0, 51.0, 50.0, 49.0, 51.0, 52.0],
        dtype=np.float64,
    )
    grid_gr = np.linspace(40.0, 70.0, 151, dtype=np.float64)
    return md, z, gr, grid_gr


def test_stage0_is_complete_and_all_future_execution_is_locked(
    train, config
):
    counts = train.validate_execution_contract(
        config,
        require_run_approval=False,
    )
    assert config["experiment"]["status"] == "stage0_fail_closed"
    assert config["implementation"]["enabled"] is True
    assert config["implementation"]["implementation_approval_received"] is True
    assert config["implementation"]["canonical_train_notebook_adopted"] is True
    assert config["implementation"]["canonical_inference_notebook_adopted"] is False
    assert config["implementation"]["kaggle_package_created"] is True
    assert config["implementation"]["train_execution_enabled"] is False
    assert config["implementation"]["inference_enabled"] is False
    assert config["implementation"]["submission_enabled"] is False
    assert config["execution"]["kaggle_push_approved"] is True
    assert config["execution"]["stage_0_execution_approved"] is True
    assert config["execution"]["stage_0_completed"] is True
    assert config["execution"]["stage_0_result"] == "stage0_fail_closed"
    assert config["execution"]["stage_0_all_gates_pass"] is False
    assert config["execution"]["stage_1_eligible_pending_separate_user_approval"] is False
    assert config["execution"]["run_stage_0"] is False
    assert config["execution"]["run_stage_1"] is False
    assert config["execution"]["run_inference"] is False
    assert config["execution"]["create_submission"] is False
    assert counts["stage_0_candidate_pf_well_runs"] == 32
    assert counts["stage_0_seed_well_trajectories"] == 4096
    assert counts["stage_0_particle_starts"] == 2048000
    assert counts["stage_0_zero_acceleration_sentinel_wells"] == 4
    assert counts["stage_1_candidate_pf_well_runs"] == 773
    assert counts["stage_1_seed_well_trajectories"] == 98944
    assert counts["stage_1_particle_starts"] == 49472000
    assert counts["control_pf_well_runs"] == 0
    assert counts["lightgbm_configs"] == 0
    assert counts["boosters"] == 0
    assert counts["hmm_well_runs"] == 0
    assert counts["beam_well_runs"] == 0
    assert counts["gpu_runs"] == 0
    with pytest.raises(RuntimeError, match="execution is disabled"):
        train.validate_execution_contract(config, require_run_approval=True)


def test_scientific_contract_pins_single_acceleration_factor(train, config):
    contract = train.validate_scientific_contract(config)
    assert contract["state"] == [
        "tvt_position",
        "u_rate",
        "u_acceleration",
    ]
    assert contract["acceleration_values"] == [-0.0005, 0.0, 0.0005]
    assert contract["acceleration_transition"] == [
        [0.92, 0.08, 0.0],
        [0.08, 0.84, 0.08],
        [0.0, 0.08, 0.92],
    ]
    assert contract["pf"]["particles"] == 500
    assert contract["pf"]["seeds"] == 128
    assert contract["pf"]["temperature"] == 5.0
    assert contract["saved_control_rerun"] is False
    assert contract["rng"]["acceleration_advances_base_stream"] is False
    assert len(contract["scientific_contract_sha256"]) == 64

    broken = deepcopy(config)
    broken["model"]["acceleration"]["transition_matrix"][1][1] = 0.83
    with pytest.raises(ValueError, match="transition changed"):
        train.validate_scientific_contract(broken)


def test_transition_boundary_fold_initial_prior_and_update_identity(train, config):
    acceleration = config["model"]["acceleration"]
    matrix = train.acceleration_transition_matrix(acceleration)
    np.testing.assert_array_equal(
        matrix,
        np.asarray(
            [
                [0.92, 0.08, 0.0],
                [0.08, 0.84, 0.08],
                [0.0, 0.08, 0.92],
            ]
        ),
    )
    np.testing.assert_allclose(matrix.sum(axis=1), 1.0, rtol=0.0, atol=1e-12)
    report = train.acceleration_transition_contract(acceleration)
    assert report["initial_probability"] == [0.0, 1.0, 0.0]
    assert report["boundary_negative_outward_mass_folded_to_stay"] is True
    assert report["boundary_positive_outward_mass_folded_to_stay"] is True
    update = train.synthetic_update_order_contract()
    assert update["pass"] is True
    assert update["minus_delta_z_identity_max_abs_error"] == 0.0


def test_fixed32_manifest_is_sha_pinned_balanced_and_prefreeze_well_only(
    train, config
):
    observed = hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest()
    assert observed == config["data"]["fixed32_manifest"]["expected_sha256"]
    manifest = pd.read_csv(MANIFEST_PATH, dtype={"well": str})
    assert len(manifest) == 32
    assert manifest["well"].nunique() == 32
    assert manifest["role"].value_counts().to_dict() == {
        "control": 16,
        "persistent": 16,
    }
    assert set(manifest["fold"].astype(int)) == set(range(5))
    wells, report = train.load_fixed32_scope(config)
    assert len(wells) == 32
    assert report["columns_read_before_freeze"] == ["well"]


def test_acceleration_seed_stream_is_stable_separate_and_particle_ordered(train):
    first = train.acceleration_seed_vector("train", "well-a", 128)
    second = train.acceleration_seed_vector("train", "well-a", 128)
    changed = train.acceleration_seed_vector("test", "well-a", 128)
    base = train.stable_seed("likpf", "train", "well-a") + np.arange(128)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, changed)
    assert not np.array_equal(first, base)
    assert (first > 0).all()
    state = int(first[0])
    draws = []
    for _ in range(4):
        state, draw = train._park_miller_uniform(state)
        draws.append(draw)
    assert all(0.0 < draw < 1.0 for draw in draws)
    assert len(set(draws)) == 4


def test_pf_input_preparation_matches_exp404_with_prefix_missing_gr(
    train, exp404
):
    horizontal = pd.DataFrame(
        {
            "MD": np.arange(12, dtype=np.float64) * 10.0,
            "Z": np.linspace(0.0, 1.1, 12),
            "GR": [
                50.0,
                np.nan,
                52.0,
                54.0,
                51.0,
                49.0,
                48.0,
                np.nan,
                53.0,
                55.0,
                54.0,
                52.0,
            ],
            "TVT_input": [
                100.0,
                100.5,
                101.0,
                101.5,
                102.0,
                102.5,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ],
        }
    )
    typewell = pd.DataFrame(
        {
            "TVT": np.linspace(95.0, 110.0, 31),
            "GR": np.linspace(45.0, 75.0, 31),
        }
    )
    expected = exp404.prepare_likelihood_pf_inputs(
        horizontal,
        typewell,
        multiplier=1.0,
        grid_step=0.2,
    )
    observed = train.prepare_likelihood_pf_inputs(
        horizontal,
        typewell,
        grid_step=0.2,
    )
    for key in (
        "eval_indices",
        "eval_md",
        "eval_z",
        "eval_gr",
        "raw_gr_observed",
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
    assert (
        observed["scale_audit"]["candidate_scale"]
        == expected["scale_audit"]["candidate_scale"]
    )


def test_zero_acceleration_kernel_is_bitwise_exp404_parity(
    train, exp404
):
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
    parent = train._pf_parent_allseeds(
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
        0.01,
    )
    zero = train._pf_persistent_acceleration_allseeds(
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
        train.acceleration_seed_vector("train", "synthetic", 4),
        np.zeros(3),
        np.asarray(
            [
                [0.92, 0.08, 0.0],
                [0.08, 0.84, 0.08],
                [0.0, 0.08, 0.92],
            ]
        ),
        0.998,
        0.002,
        0.005,
        0.1,
        0.001,
        0.5,
        4.5,
        0.01,
    )
    for index in range(5):
        assert np.array_equal(parent[index], expected[index])
        assert np.array_equal(zero[index], expected[index])


def test_persistent_kernel_preserves_probability_and_state_count_contract(train):
    md, z, gr, grid_gr = synthetic_pf_inputs()
    output = train._pf_persistent_acceleration_allseeds(
        md,
        z,
        gr,
        grid_gr,
        90.0,
        0.2,
        20.0,
        100.0,
        0.01,
        64,
        8,
        4321,
        train.acceleration_seed_vector("train", "persistent", 8),
        np.asarray([-0.0005, 0.0, 0.0005]),
        np.asarray(
            [
                [0.92, 0.08, 0.0],
                [0.08, 0.84, 0.08],
                [0.0, 0.08, 0.92],
            ]
        ),
        0.998,
        0.002,
        0.005,
        0.1,
        0.001,
        0.5,
        4.5,
        0.01,
    )
    prediction, log_likelihood = output[:2]
    acceleration_mass = output[5]
    effective_sample_size = output[8]
    state_counts = output[9]
    assert np.isfinite(prediction).all()
    assert np.isfinite(log_likelihood).all()
    assert np.isfinite(effective_sample_size).all()
    assert (effective_sample_size > 0.0).all()
    np.testing.assert_allclose(
        acceleration_mass.sum(axis=2),
        1.0,
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_array_equal(
        state_counts.sum(axis=2),
        np.full((8, len(md)), 64),
    )
    assert int(state_counts[:, :, 0].sum()) > 0
    assert int(state_counts[:, :, 2].sum()) > 0


def test_leakage_ledger_requires_all_wells_frozen(train):
    ledger = train.LeakageLedger(expected_wells=2)
    ledger.freeze("a")
    with pytest.raises(RuntimeError, match="before all fixed32 artifacts"):
        ledger.record_truth(10)
    ledger.freeze("b")
    ledger.record_control(20)
    ledger.record_role_fold(2)
    ledger.record_episode(1)
    report = ledger.report()
    assert report["all_frozen"] is True
    assert report["before_freeze"]["truth_rows"] == 10
    assert report["after_freeze"]["control_rows"] == 20


def test_compact_source_is_not_thin_and_canonical_train_is_adopted(config):
    source = SOURCE.read_text()
    for heading in (
        "Exp404 likelihood-PF input preparation",
        "Persistent acceleration transition and likelihood-PF kernel",
        "Zero-acceleration exp404 parity",
        "Target-free candidate generation and freeze",
        "Truth-late mechanism readout",
        "Generated artifacts",
    ):
        assert heading in source
    assert "Path(__file__)" not in source
    assert "from settings import" not in source
    assert "def _pf_persistent_acceleration_allseeds(" in source
    assert "def run_stage0(" in source
    assert COMPACT_TRAIN.exists()
    assert CANONICAL_TRAIN.exists()
    canonical = json.loads(CANONICAL_TRAIN.read_text())
    canonical_text = "\n".join(
        "".join(cell.get("source", [])) for cell in canonical["cells"]
    )
    assert "def _pf_persistent_acceleration_allseeds(" in canonical_text
    assert "def run_stage0(" in canonical_text
    assert len(canonical["cells"]) >= 20
    assert config["implementation"]["canonical_train_notebook_adopted"] is True
