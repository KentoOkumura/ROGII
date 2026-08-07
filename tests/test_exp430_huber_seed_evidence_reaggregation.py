from __future__ import annotations

import copy
import hashlib
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
EXP = "exp430_huber_seed_evidence_reaggregation"
EXP_DIR = ROOT / "experiments" / EXP
TRAIN_SOURCE = EXP_DIR / f"{EXP}_compact_selfcontained_train.py"
INFERENCE_SOURCE = EXP_DIR / f"{EXP}_compact_selfcontained_inference.py"
PARENT_SOURCE = (
    ROOT
    / "experiments"
    / "exp404_scale5_sigma_gr_likelihood_pf_ablation"
    / "exp404_scale5_sigma_gr_likelihood_pf_ablation_compact_selfcontained_train.py"
)
PREFLIGHT_ASSET = EXP_DIR / "assets" / f"{EXP}_preflight_wells.csv"


def load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def train() -> ModuleType:
    previous = os.environ.get("EXP430_IMPORT_ONLY")
    os.environ["EXP430_IMPORT_ONLY"] = "1"
    try:
        return load_module(TRAIN_SOURCE, "exp430_train_contract")
    finally:
        if previous is None:
            os.environ.pop("EXP430_IMPORT_ONLY", None)
        else:
            os.environ["EXP430_IMPORT_ONLY"] = previous


@pytest.fixture(scope="module")
def inference() -> ModuleType:
    return load_module(INFERENCE_SOURCE, "exp430_inference_contract")


@pytest.fixture(scope="module")
def parent() -> ModuleType:
    return load_module(PARENT_SOURCE, "exp404_parent_for_exp430")


@pytest.fixture(scope="module")
def config() -> dict:
    return yaml.safe_load((EXP_DIR / "config.yaml").read_text())


def test_frozen_contract_and_execution_counts(
    train: ModuleType,
    config: dict,
) -> None:
    contract = train.validate_scientific_contract(config)
    counts = contract["execution_counts"]

    assert contract["route"] == "pf_beam"
    assert contract["parent"] == "exp404_scale5_sigma_gr_likelihood_pf_ablation"
    assert contract["primary_control"] == "gaussian_matched"
    assert contract["primary_candidate"] == "huber_delta_1p345"
    assert contract["pf"]["particles"] == 500
    assert contract["pf"]["seeds"] == 128
    assert contract["pf"]["trajectory_storage_dtype"] == "float64"
    assert contract["evidence"]["temperature"] == 5.0
    assert contract["evidence"]["huber_delta_1p345"]["delta"] == 1.345
    assert counts["technical_preflight_pf_well_runs"] == 4
    assert counts["technical_preflight_seed_well_trajectories"] == 512
    assert counts["technical_preflight_particle_starts"] == 256_000
    assert counts["full_pf_well_runs"] == 773
    assert counts["full_seed_well_trajectories"] == 98_944
    assert counts["full_particle_starts"] == 49_472_000
    assert counts["full_shards"] == 4
    assert counts["parent_independent_full_reruns"] == 0
    assert counts["lightgbm_configs"] == 0
    assert counts["trained_folds"] == 0
    assert counts["boosters"] == 0
    assert counts["models"] == 0
    assert counts["gpu_runs"] == 0
    assert len(contract["scientific_contract_sha256"]) == 64
    stage = train.selected_stage(config)
    assert stage in {None, "preflight", "full_shard", "merge"}
    if stage is None:
        with pytest.raises(RuntimeError, match="stage is not selected"):
            train.validate_scientific_contract(config, require_run_approval=True)
    else:
        approved_contract = train.validate_scientific_contract(
            config,
            require_run_approval=True,
        )
        assert approved_contract["scientific_contract_sha256"] == (
            contract["scientific_contract_sha256"]
        )


def test_contract_rejects_grids_and_parent_pf_changes(
    train: ModuleType,
    config: dict,
) -> None:
    broken = copy.deepcopy(config)
    broken["model"]["evidence"]["huber_delta_1p345"]["delta"] = 1.5
    with pytest.raises(ValueError, match="scientific contract mismatch"):
        train.validate_scientific_contract(broken)

    broken = copy.deepcopy(config)
    broken["model"]["evidence"]["temperature"] = 8.0
    with pytest.raises(ValueError, match="scientific contract mismatch"):
        train.validate_scientific_contract(broken)

    broken = copy.deepcopy(config)
    broken["model"]["pf"]["particles"] = 1_000
    with pytest.raises(ValueError, match="scientific contract mismatch"):
        train.validate_scientific_contract(broken)

    broken = copy.deepcopy(config)
    broken["model"]["active_scientific_variants"].append("forbidden_grid")
    with pytest.raises(ValueError, match="active_scientific_variants"):
        train.validate_scientific_contract(broken)


def test_stable_seed_matches_exp404_parent(
    train: ModuleType,
    parent: ModuleType,
) -> None:
    for well in ("24d8997e", "e7818f7a", "fd710aea", "ea41324e"):
        assert train.stable_seed("likpf", "train", well) == parent.stable_seed(
            "likpf",
            "train",
            well,
        )
        expected = (
            int(
                hashlib.sha256(f"likpf::train::{well}".encode()).hexdigest()[:16],
                16,
            )
            % 2_147_483_647
            + 1
        )
        assert train.stable_seed("likpf", "train", well) == expected


def test_pf_kernel_has_exact_exp404_rng_and_prediction_parity(
    train: ModuleType,
    parent: ModuleType,
) -> None:
    args = (
        np.arange(1.0, 10.0, dtype=np.float64),
        np.linspace(0.0, 0.8, 9, dtype=np.float64),
        np.asarray([50.0, 51.0, 53.0, 54.0, 52.0, 49.0, 48.0, 50.0, 51.0]),
        np.linspace(40.0, 70.0, 151, dtype=np.float64),
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
    expected = parent._pf_lik_allseeds(*args)
    observed = train._pf_lik_allseeds(*args)

    assert len(observed) == len(expected) == 5
    for actual, control in zip(observed, expected, strict=True):
        np.testing.assert_array_equal(actual, control)


def test_gaussian_and_huber_formulas_are_fixed(
    train: ModuleType,
) -> None:
    trajectories = np.asarray(
        [
            [0.0, 1.0],
            [2.0, 1.0],
        ],
        dtype=np.float64,
    )
    prepared = {
        "grid_gr": np.asarray([0.0, 10.0, 20.0], dtype=np.float64),
        "grid_minimum": 0.0,
        "grid_step": 1.0,
        "eval_gr": np.asarray([0.0, 12.0], dtype=np.float64),
        "gr_scale": 2.0,
    }
    gaussian, huber = train.trajectory_residual_scores(
        trajectories,
        prepared,
        delta=1.345,
    )

    np.testing.assert_allclose(gaussian, np.asarray([-0.5, -50.5]))
    expected_large_huber = -(
        1.345 * 10.0 - 0.5 * 1.345**2 + 0.5
    )
    np.testing.assert_allclose(
        huber,
        np.asarray([-0.5, expected_large_huber]),
    )


def test_reaggregation_uses_one_unchanged_trajectory_bank(
    train: ModuleType,
) -> None:
    trajectories = np.asarray(
        [
            [100.0, 101.0, 102.0],
            [110.0, 111.0, 112.0],
            [90.0, 91.0, 92.0],
        ],
        dtype=np.float64,
    )
    before = trajectories.copy()
    gaussian = np.asarray([-1.0, -3.0, -4.0])
    huber = np.asarray([-2.0, -2.1, -2.2])
    parent_scores = np.asarray([-1.5, -4.0, -3.0])
    predictions, weights = train.aggregate_frozen_trajectories(
        trajectories,
        gaussian,
        huber,
        parent_scores,
        temperature=5.0,
    )

    np.testing.assert_array_equal(trajectories, before)
    assert set(predictions) == {
        "gaussian_matched",
        "huber_delta_1p345",
        "arithmetic_mean",
        "parent_gaussian_marginal_replay",
    }
    assert set(weights) == {
        "gaussian_matched",
        "huber_delta_1p345",
        "parent_gaussian_marginal_replay",
    }
    for value in weights.values():
        assert abs(float(value.sum()) - 1.0) <= 1.0e-15
        assert np.isfinite(value).all()
    np.testing.assert_allclose(
        predictions["arithmetic_mean"],
        trajectories.mean(axis=0),
    )


def test_parent_parity_uses_exp404_float32_storage_semantics(
    train: ModuleType,
) -> None:
    replay_binary_value = np.asarray([11183.765], dtype=np.float32).astype(np.float64)
    parent_csv_value = np.asarray([11183.765], dtype=np.float64)

    assert float(abs(replay_binary_value[0] - parent_csv_value[0])) > 1.0e-5
    np.testing.assert_array_equal(
        train.float32_storage_values(replay_binary_value),
        train.float32_storage_values(parent_csv_value),
    )


def test_lpt_shards_are_deterministic_and_seed_order_independent(
    train: ModuleType,
) -> None:
    manifest = pd.DataFrame(
        {
            "well_id": ["w3", "w1", "w4", "w0", "w2"],
            "rows": [10, 10, 10, 10, 10],
            "prefix_rows": [2, 2, 2, 2, 2],
            "suffix_rows": [9, 7, 5, 3, 1],
        }
    )
    first = train.assign_lpt_shards(manifest, 4).sort_values("well_id")
    second = train.assign_lpt_shards(
        manifest.sample(frac=1.0, random_state=7),
        4,
    ).sort_values("well_id")
    pd.testing.assert_series_equal(
        first["shard_index"].reset_index(drop=True),
        second["shard_index"].reset_index(drop=True),
    )
    first_seed = {
        well: train.stable_seed("likpf", "train", well)
        for well in first["well_id"]
    }
    second_seed = {
        well: train.stable_seed("likpf", "train", well)
        for well in reversed(first["well_id"].tolist())
    }
    assert first_seed == second_seed


def test_truth_access_ledger_fails_closed(
    train: ModuleType,
) -> None:
    ledger = train.TruthAccessLedger()
    with pytest.raises(RuntimeError, match="requires frozen"):
        ledger.require_frozen()
    ledger.fold_rows_before_freeze = 1
    with pytest.raises(RuntimeError, match="before freeze"):
        ledger.mark_frozen()

    clean = train.TruthAccessLedger()
    clean.mark_frozen()
    clean.require_frozen()
    assert clean.report()["prediction_frozen"] is True
    assert all(value == 0 for value in clean.report()["before_freeze"].values())


def test_fixed_preflight_asset_is_sha_first_and_pinned(
    train: ModuleType,
    config: dict,
) -> None:
    asset = pd.read_csv(PREFLIGHT_ASSET, dtype={"well_id": str})
    assert asset["well_id"].tolist() == [
        "24d8997e",
        "e7818f7a",
        "fd710aea",
        "ea41324e",
    ]
    assert asset["sha256_order"].tolist() == [0, 1, 2, 3]
    assert train.sha256_path(PREFLIGHT_ASSET) == config["data"]["preflight_wells"][
        "expected_sha256"
    ]
    observed = sorted(
        asset["well_id"],
        key=lambda value: (hashlib.sha256(value.encode()).hexdigest(), value),
    )
    assert observed == asset["well_id"].tolist()


def test_inference_is_explicitly_disabled(
    inference: ModuleType,
    config: dict,
) -> None:
    report = inference.validate_inference_disabled(config)
    assert report["status"] == (
        "inference_disabled_pending_train_gate_and_separate_approval"
    )
    assert all(report["checks"].values())

    broken = copy.deepcopy(config)
    broken["inference"]["enabled"] = True
    with pytest.raises(RuntimeError, match="inference contract changed"):
        inference.validate_inference_disabled(broken)


def test_jupytext_sources_are_self_contained_and_metrics_json_is_valid() -> None:
    train_source = TRAIN_SOURCE.read_text()
    inference_source = INFERENCE_SOURCE.read_text()
    assert train_source.count("# %% [markdown]") >= 10
    assert "## Contents" in train_source
    assert "def _pf_lik_allseeds(" in train_source
    assert "def trajectory_residual_scores(" in train_source
    assert "def run_preflight_stage(" in train_source
    assert "def run_full_shard_stage(" in train_source
    assert "def run_merge_stage(" in train_source
    assert "from exp430" not in train_source
    assert "__file__" not in train_source
    assert "## Contents" in inference_source
    assert "from exp430" not in inference_source
    assert "__file__" not in inference_source

    metrics = __import__("json").loads((EXP_DIR / "metrics.json").read_text())
    assert metrics["experiment"] == EXP
    assert metrics["route"] == "pf_beam"
