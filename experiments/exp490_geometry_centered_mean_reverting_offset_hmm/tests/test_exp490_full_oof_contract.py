from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
SOURCE = (
    ROOT
    / "experiments"
    / "exp490_geometry_centered_mean_reverting_offset_hmm"
    / "exp490_geometry_centered_mean_reverting_offset_hmm_train_aggregate.py"
)


def load_module():
    name = "exp490_full_oof_train_contract"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_stable_full_well_shard_is_deterministic_and_bounded() -> None:
    module = load_module()
    wells = ["00000000", "11d0f5ac", "ffefef30"]
    first = [module.stable_full_well_shard(well) for well in wells]
    second = [module.stable_full_well_shard(well) for well in wells]
    assert first == second
    assert all(0 <= value < 4 for value in first)


def test_stage1_execution_contract_is_one_variant_773_wells_zero_control_reruns() -> None:
    module = load_module()
    stage1_config = copy.deepcopy(module.CONFIG)
    stage1_config["execution"]["run_inference"] = False
    stage1_config["execution"]["create_submission"] = False
    contract = module.validate_execution_contract(
        stage1_config,
        require_run_authorization=True,
    )
    assert contract["scientific_variants"] == 1
    assert contract["candidate_hmm_well_runs"] == 773
    assert contract["saved_parent_hmm_well_runs"] == 0
    assert contract["operational_cpu_shards"] == 4
    assert contract["merge_hmm_well_runs"] == 0
    assert contract["model_configs"] == 0
    assert contract["trained_folds"] == 0
    assert contract["boosters"] == 0


def test_pinned_shard_allocation_contract_totals_full_oof() -> None:
    module = load_module()
    expected = pd.DataFrame(
        module.get_nested(module.CONFIG, "data.stage_1_shards.expected")
    )
    assert expected["wells"].sum() == 773
    assert expected["rows"].sum() == 3_783_989
    assert expected["shard_index"].tolist() == [0, 1, 2, 3]


def test_full_source_preserves_original_physical_scientific_contract() -> None:
    module = load_module()
    contract = module.validate_scientific_contract(module.CONFIG)
    assert contract["candidate_variants"] == 1
    assert contract["coordinate"]["rate_center"] == "0.998 * rho_t * q_previous"
    assert contract["coordinate"]["offset_center"] == (
        "rho_t * residual_offset_previous + q_t * positive_dMD_t"
    )
    assert contract["rho"]["half_life_segments"] == 1.0
