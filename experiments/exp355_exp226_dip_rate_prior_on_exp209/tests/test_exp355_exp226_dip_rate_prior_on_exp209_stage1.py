from __future__ import annotations

import importlib.util
import os
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = ROOT / "experiments" / "exp355_exp226_dip_rate_prior_on_exp209"
TRAIN_SOURCE = EXP_DIR / (
    "exp355_exp226_dip_rate_prior_on_exp209_stage1_compact_selfcontained_train.py"
)


def load_module(path: Path, name: str):
    previous = os.environ.get("EXP355_IMPORT_ONLY")
    os.environ["EXP355_IMPORT_ONLY"] = "1"
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        if previous is None:
            os.environ.pop("EXP355_IMPORT_ONLY", None)
        else:
            os.environ["EXP355_IMPORT_ONLY"] = previous


@pytest.fixture(scope="module")
def train():
    return load_module(TRAIN_SOURCE, "exp355_stage1_train_test")


@pytest.fixture(scope="module")
def config():
    value = yaml.safe_load((EXP_DIR / "config.yaml").read_text())
    assert isinstance(value, dict)
    return value


def test_stage1_contract_is_one_cpu_candidate_with_no_control_rerun(
    train, config
) -> None:
    counts = train.validate_scientific_contract(config)
    assert counts == {
        "scientific_variants": 1,
        "reporting_folds": 5,
        "hmm_well_runs": 773,
        "model_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
        "gpu": False,
        "parent_control_retraining": False,
    }
    with pytest.raises(RuntimeError, match="run_stage_1 must be explicitly enabled"):
        train.validate_scientific_contract(config, require_run_approval=True)

    approved = deepcopy(config)
    approved["execution"]["run_stage_1"] = True
    train.validate_scientific_contract(approved, require_run_approval=True)

    unapproved = deepcopy(approved)
    unapproved["execution"]["kaggle_push_approved"] = False
    with pytest.raises(RuntimeError, match="package/push/run is not approved"):
        train.validate_scientific_contract(unapproved, require_run_approval=True)

    disabled = deepcopy(approved)
    disabled["execution"]["run_stage_1"] = False
    with pytest.raises(RuntimeError, match="run_stage_1 must be explicitly enabled"):
        train.validate_scientific_contract(disabled, require_run_approval=True)


def test_user_override_is_explicit_and_does_not_enable_inference(config) -> None:
    override = config["model"]["stage_1"]["user_override"]
    assert override["approved"] is True
    assert override["overridden_failed_gate"] == "stage0_worst_well_regression_guard"
    assert set(override["does_not_approve"]) == {
        "parameter_rescue",
        "inference",
        "submission",
    }
    assert config["execution"]["run_inference"] is False
    assert config["execution"]["create_submission"] is False


def test_stage0_schedule_sha_is_frozen_into_stage1_contract(config) -> None:
    stage1 = config["model"]["stage_1"]
    assert stage1["expected_stage0_schedule_logical_sha256"] == (
        "53f9d42bcca0f5596568971b5da6c440114922d0a25b5622592e1b7b50774c85"
    )
    assert stage1["expected_stage0_geometry_ledger_logical_sha256"] == (
        "b527d3401e2d730ec883681051c476c929a428e7fc28ed88fff3091045915a39"
    )


def test_residual_hmm_absorbs_frozen_mu_into_effective_dz(train, config) -> None:
    horizontal = pd.DataFrame(
        {
            "MD": np.arange(6, dtype=float),
            "Z": np.zeros(6, dtype=float),
            "GR": np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0]),
            "TVT_input": np.array(
                [100.0, 100.5, 101.0, 101.5, np.nan, np.nan]
            ),
        }
    )
    typewell = pd.DataFrame(
        {
            "TVT": np.linspace(90.0, 110.0, 41),
            "GR": np.linspace(5.0, 25.0, 41),
        }
    )
    schedule = pd.DataFrame(
        {
            "row_idx": np.array([4, 5], dtype=np.int32),
            "mu_rate": np.array([0.5, 0.7], dtype=float),
        }
    )
    prepared = train.prepare_hmm_inputs(horizontal, typewell, schedule, config)
    np.testing.assert_allclose(prepared["dm"], [1.0, 1.0])
    np.testing.assert_allclose(prepared["effective_dz"], [-0.5, -0.7])
    assert prepared["initial_residual_rate"] == pytest.approx(0.0)
    np.testing.assert_allclose(
        prepared["rates"],
        np.linspace(-0.10, 0.10, 41),
    )


def test_k16_schedule_uses_exp226_row_position_partition(train) -> None:
    segment_id = train.k16_segment_ids(35, 16)
    expected = np.searchsorted(
        np.linspace(0.0, 35.0, 17)[1:],
        np.arange(1.0, 36.0),
        side="left",
    )
    np.testing.assert_array_equal(segment_id, expected)
