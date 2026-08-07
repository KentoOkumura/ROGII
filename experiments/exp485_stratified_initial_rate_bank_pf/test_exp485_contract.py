from __future__ import annotations

import importlib.util
import os
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SOURCE = (
    Path(__file__).parent
    / "exp485_stratified_initial_rate_bank_pf_compact_selfcontained_train.py"
)
os.environ["EXP485_IMPORT_ONLY"] = "1"
SPEC = importlib.util.spec_from_file_location("exp485_train", SOURCE)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_scientific_contract_matches_config() -> None:
    contract = MODULE.validate_scientific_contract(
        MODULE.load_experiment_config(Path(__file__).parent),
        require_run_approval=False,
    )
    assert contract["changed_factor"]["windows_rows"] == [30, 32, 64, 128, 256]
    assert len(contract["scientific_contract_sha256"]) == 64


def test_equal_strata_interleave_is_exact() -> None:
    component = MODULE.stratified_component_ids(500)
    assert component[:15].tolist() == [0, 1, 2, 3, 4] * 3
    np.testing.assert_array_equal(np.bincount(component), np.full(5, 100))


def test_rate_formula_and_duplicate_centers_are_preserved() -> None:
    frame = pd.DataFrame(
        {
            "MD": np.arange(10, dtype=np.float64),
            "Z": np.arange(10, dtype=np.float64) * 0.25,
            "TVT_input": np.arange(10, dtype=np.float64) * 0.75,
            "GR": np.linspace(40.0, 50.0, 10),
        }
    )
    centers, diagnostics = MODULE.initial_rate_bank(frame)
    np.testing.assert_array_equal(centers, np.ones(5))
    assert len(diagnostics) == 5
    assert diagnostics["unique_center_count"].eq(1).all()


def test_rate_fallback_requires_three_valid_steps() -> None:
    frame = pd.DataFrame(
        {
            "MD": [0.0, 1.0, 2.0],
            "Z": [0.0, 0.1, 0.2],
            "TVT_input": [0.0, 0.9, 1.8],
        }
    )
    rate, _, valid_steps, used_fallback = MODULE.robust_initial_rate(frame, 30)
    assert valid_steps == 2
    assert used_fallback
    assert rate == 0.0


def test_short_suffix_checkpoints_are_well_defined() -> None:
    np.testing.assert_array_equal(
        MODULE.checkpoint_indices(10),
        np.array([0, 9, 9, 9, 9]),
    )


def test_duplicated_center_is_bitwise_exp404_parity() -> None:
    report = MODULE.duplicated_center_exp404_parity_contract()
    assert report["prediction_bitwise_equal"]
    assert report["loglik_bitwise_equal"]
    assert report["pass"]


def test_stable_seed_contract() -> None:
    first = MODULE.stable_seed("likpf", "train", "well-a")
    second = MODULE.stable_seed("likpf", "train", "well-a")
    other = MODULE.stable_seed("likpf", "train", "well-b")
    assert first == second
    assert first != other


def test_stage1_is_separately_approved_under_recorded_runtime_exception() -> None:
    config = MODULE.load_experiment_config(Path(__file__).parent)
    assert config["execution"]["run_stage_0"] is False
    assert config["execution"]["run_stage_1"] is False
    assert config["execution"]["stage_1_completed"] is True
    assert (
        config["execution"]["stage_1_result"]
        == "stage1_gate_failed_terminal_close"
    )
    execution_config = deepcopy(config)
    execution_config["execution"]["run_stage_1"] = True
    counts = MODULE.validate_execution_contract(
        execution_config,
        require_run_approval=True,
    )
    assert config["execution"]["stage_1_execution_approved"] is True
    assert config["stage_0_result"]["all_non_runtime_gates_passed"] is True
    assert config["stage_0_result"]["original_runtime_gate_passed"] is False
    assert config["stage_0_result"]["runtime_exception"]["approved"] is True
    assert config["data"]["stage1_resume"]["enabled"] is True
    assert config["execution"]["stage_1_resume_from_version_2_approved"] is True
    assert counts["stage_1_candidate_pf_well_runs"] == 773
    assert counts["stage_1_seed_well_trajectories"] == 98_944
    assert counts["stage_1_particle_starts"] == 49_472_000
    denied = deepcopy(execution_config)
    denied["execution"]["stage_1_execution_approved"] = False
    with pytest.raises(RuntimeError, match="Stage 1 is not approved"):
        MODULE.validate_execution_contract(denied, require_run_approval=True)
    resume_denied = deepcopy(execution_config)
    resume_denied["execution"]["stage_1_resume_from_version_2_approved"] = False
    with pytest.raises(RuntimeError, match="Stage 1 resume is not approved"):
        MODULE.validate_execution_contract(
            resume_denied,
            require_run_approval=True,
        )


def test_selected_csv_content_sha_is_storage_representation_independent(
    tmp_path: Path,
) -> None:
    frame = pd.DataFrame(
        {
            "id": ["w0_0", "w0_1"],
            "hmm_mean_tvt": [1.25, 2.5],
            "unused": [100, 200],
        }
    )
    plain = tmp_path / "hmm.csv"
    zipped = tmp_path / "hmm.csv.gz"
    frame.to_csv(plain, index=False)
    frame.to_csv(zipped, index=False, compression="gzip")
    _, plain_report = MODULE.read_selected_csv_with_content_sha(
        plain,
        ["id", "hmm_mean_tvt"],
        numeric_columns=["hmm_mean_tvt"],
        chunksize=1,
    )
    _, zipped_report = MODULE.read_selected_csv_with_content_sha(
        zipped,
        ["id", "hmm_mean_tvt"],
        numeric_columns=["hmm_mean_tvt"],
        chunksize=1,
    )
    assert plain_report["rows"] == 2
    assert (
        plain_report["selected_columns_sha256"]
        == zipped_report["selected_columns_sha256"]
    )
    assert MODULE.sha256_csv_payload(plain) == MODULE.sha256_decompressed_csv(
        zipped
    )


def test_stage1_gate_preserves_original_runtime_failure_and_applies_exception() -> None:
    config = deepcopy(MODULE.load_experiment_config(Path(__file__).parent))
    rows = 773
    wells = [f"w{index:03d}" for index in range(rows)]
    frame = pd.DataFrame(
        {
            "id": [f"{well}_0" for well in wells],
            "well_id": wells,
            "row_idx": np.zeros(rows, dtype=np.int64),
            "suffix_offset": np.zeros(rows, dtype=np.int64),
            "md_since": np.where(np.arange(rows) % 2 == 0, 1200.0, 100.0),
            "raw_gr_observed": np.arange(rows) % 2 == 0,
            "well_missing_fraction": np.where(np.arange(rows) % 2 == 0, 0.1, 0.4),
            MODULE.PRIMARY_CANDIDATE: np.zeros(rows, dtype=np.float32),
            MODULE.PRIMARY_CONTROL: np.ones(rows, dtype=np.float32),
            "saved_exp209_hmm": np.zeros(rows),
            "candidate_hmm_50_50": np.zeros(rows),
            "control_hmm_50_50": np.full(rows, 0.5),
            "true_tvt": np.zeros(rows),
            "fold": np.arange(rows) % 5,
            "hidden_like_spatial": np.arange(rows) < 200,
            "hidden_like_typewell_purged": (np.arange(rows) >= 200)
            & (np.arange(rows) < 400),
        }
    )
    primary, by_well, blend = MODULE.build_stage1_metric_outputs(frame)
    audit = pd.DataFrame(
        {
            "well_id": wells,
            "status": ["ok"] * rows,
            "pf_well_runs": np.ones(rows, dtype=np.int64),
            "seed_well_trajectories": np.full(rows, 128, dtype=np.int64),
            "particle_starts": np.full(rows, 64_000, dtype=np.int64),
            "initial_component_counts_contract": [True] * rows,
            "rate_unique_center_count": np.full(rows, 2, dtype=np.int64),
        }
    )
    config["validation"]["expected_rows"] = rows
    config["validation"]["expected_wells"] = rows
    config["validation"]["primary_control_rmse_ft"] = 1.0
    config["validation"]["fixed_hmm_pf_50_50_control_rmse_ft"] = 0.5
    config["data"]["expected_raw_well_identity_sha256"] = "a" * 64
    gate = MODULE.evaluate_stage1_gate(
        config,
        frame,
        pd.DataFrame(index=np.arange(rows * 5)),
        pd.DataFrame(index=np.arange(rows * 25)),
        audit,
        {"sha_readback": {"pass": True}},
        primary,
        by_well,
        blend,
        {
            "before_freeze": {
                "truth_rows": 0,
                "control_rows": 0,
                "fold_rows": 0,
                "hidden_like_rows": 0,
            }
        },
        {"content_sha256": "a" * 64},
        40_000.0,
        1.0,
    )
    technical = gate["technical_gate"]
    assert technical["original_runtime_gate_passed"] is False
    assert technical["runtime_user_exception_applied"] is True
    assert technical["passed"] is True
    assert gate["passed"] is True


def test_stage1_freeze_roundtrip_preserves_prediction_sha(tmp_path: Path) -> None:
    prediction = pd.DataFrame(
        {
            "id": ["w000_1"],
            "well_id": ["w000"],
            "row_idx": np.array([1], dtype=np.int64),
            "suffix_offset": np.array([0], dtype=np.int64),
            "last_known_tvt": np.array([10.0]),
            "md_since": np.array([1.0]),
            "raw_gr_observed": np.array([True]),
            "well_missing_fraction": np.array([0.0]),
            MODULE.PRIMARY_CANDIDATE: np.array([0.12345679], dtype=np.float32),
        }
    )
    rate_bank = pd.DataFrame(
        {
            "well_id": ["w000"] * 5,
            "component_index": np.arange(5),
            "window_rows": [30, 32, 64, 128, 256],
            "center_value": np.arange(5, dtype=float),
            "valid_steps": np.full(5, 10),
            "used_fallback": [False] * 5,
        }
    )
    components = pd.DataFrame(
        {
            "well_id": ["w000"] * 25,
            "checkpoint": np.repeat(MODULE.CHECKPOINT_LABELS, 5),
            "checkpoint_row": np.repeat(np.arange(5), 5),
            "component_index": np.tile(np.arange(5), 5),
            "filtered_posterior_mass_mean": np.full(25, 0.2),
            "surviving_particle_count_mean": np.full(25, 100.0),
        }
    )
    frozen_well = MODULE.FrozenWell(
        well_id="w000",
        prediction=prediction,
        rate_bank=rate_bank,
        component_ledger=components,
        audit={"well_id": "w000", "status": "ok"},
    )
    ledger = MODULE.LeakageLedger(expected_wells=1)
    _, _, _, _, frozen = MODULE.freeze_target_free_outputs(
        [frozen_well],
        tmp_path,
        stage="stage1",
        expected_rows=1,
        expected_wells=1,
        ledger=ledger,
    )
    assert frozen["sha_readback"]["pass"] is True
    assert ledger.all_frozen is True
