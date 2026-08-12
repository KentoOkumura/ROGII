from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
EXP = "exp483_huber_gr_filtering_likelihood_pf"
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


def load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def train() -> ModuleType:
    previous = os.environ.get("EXP483_IMPORT_ONLY")
    os.environ["EXP483_IMPORT_ONLY"] = "1"
    try:
        return load_module(SOURCE, "exp483_train_test")
    finally:
        if previous is None:
            os.environ.pop("EXP483_IMPORT_ONLY", None)
        else:
            os.environ["EXP483_IMPORT_ONLY"] = previous


@pytest.fixture(scope="module")
def exp404() -> ModuleType:
    return load_module(EXP404_SOURCE, "exp404_parent_for_exp483")


@pytest.fixture(scope="module")
def config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())


def synthetic_pf_inputs() -> tuple[np.ndarray, ...]:
    md = np.arange(1.0, 13.0, dtype=np.float64)
    z = np.linspace(0.0, 1.1, len(md), dtype=np.float64)
    gr = np.asarray(
        [50.0, 52.0, 90.0, 53.0, 51.0, 10.0, 49.0, 51.0, 52.0, 54.0, 55.0, 53.0],
        dtype=np.float64,
    )
    grid_gr = np.linspace(40.0, 70.0, 151, dtype=np.float64)
    return md, z, gr, grid_gr


def test_completed_stage1_is_locked_but_approval_contract_remains_valid(
    train: ModuleType,
    config: dict,
) -> None:
    counts = train.validate_execution_contract(config)
    assert config["experiment"]["status"] == "stage1_gate_failed_terminal_close"
    assert config["implementation"]["enabled"] is True
    assert config["implementation"]["implementation_approval_received"] is True
    assert config["implementation"]["jupytext_source_created"] is True
    assert config["implementation"]["tests_created"] is True
    assert config["implementation"]["canonical_train_notebook_adopted"] is True
    assert config["implementation"]["kaggle_package_created"] is True
    assert config["implementation"]["inference_enabled"] is False
    assert config["implementation"]["submission_enabled"] is False
    assert config["execution"]["kaggle_push_approved"] is True
    assert config["execution"]["stage_0_execution_approved"] is True
    assert config["execution"]["stage_1_execution_approved"] is True
    assert config["execution"]["run_stage_0"] is False
    assert config["execution"]["run_stage_1"] is False
    assert config["execution"]["stage_0_kernel_status"] == "COMPLETE"
    assert config["execution"]["stage_1_kernel_status"] == "COMPLETE"
    assert config["stage_0_result"]["all_technical_gates_passed"] is True
    assert config["stage_0_result"]["technical_gates_passed"] == 10
    assert config["stage_0_result"]["decision"]["stage_1_run"] is True
    assert config["stage_0_result"]["decision"]["stage_1_approved"] is True
    assert counts["stage_0_candidate_pf_well_runs"] == 32
    assert counts["stage_0_seed_well_trajectories"] == 4096
    assert counts["stage_0_particle_starts"] == 2048000
    assert counts["stage_1_candidate_pf_well_runs"] == 773
    assert counts["stage_1_seed_well_trajectories"] == 98944
    assert counts["stage_1_particle_starts"] == 49472000
    assert counts["control_pf_well_runs"] == 0
    assert counts["lightgbm_configs"] == 0
    assert counts["boosters"] == 0
    assert counts["hmm_well_runs"] == 0
    assert counts["beam_well_runs"] == 0
    assert counts["gpu_runs"] == 0
    assert config["stage_1_result"]["technical_gate"]["passed"] is True
    assert config["stage_1_result"]["scientific_gate"]["passed"] is False
    approved_run = deepcopy(config)
    approved_run["execution"]["run_stage_1"] = True
    train.validate_execution_contract(approved_run, require_run_approval=True)
    denied_stage1 = deepcopy(config)
    denied_stage1["execution"]["run_stage_1"] = True
    denied_stage1["execution"]["stage_1_execution_approved"] = False
    with pytest.raises(RuntimeError, match="not approved"):
        train.validate_execution_contract(denied_stage1, require_run_approval=True)
    simultaneous = deepcopy(config)
    simultaneous["execution"]["run_stage_0"] = True
    simultaneous["execution"]["run_stage_1"] = True
    with pytest.raises(ValueError, match="exactly one"):
        train.validate_execution_contract(simultaneous, require_run_approval=True)


def test_scientific_contract_pins_only_fixed_huber_filtering_factor(
    train: ModuleType,
    config: dict,
) -> None:
    contract = train.validate_scientific_contract(config)
    changed = contract["changed_factor"]
    fixed = contract["fixed_pf"]
    assert changed["family"] == "huber"
    assert changed["delta"] == 1.345
    assert changed["additional_clip"] is None
    assert changed["application_scope"] == "every_finite_gr_particle_update"
    assert fixed["particles"] == 500
    assert fixed["seeds"] == 128
    assert fixed["primary_seed_weighting_temperature"] == 5.0
    assert fixed["gr_scale_multiplier"] == 1.0
    assert contract["saved_control"]["rerun"] is False
    assert len(contract["scientific_contract_sha256"]) == 64

    broken = deepcopy(config)
    broken["model"]["changed_factor"]["delta"] = 1.5
    with pytest.raises(ValueError, match="delta"):
        train.validate_scientific_contract(broken)


def test_huber_formula_is_gaussian_inside_delta_linear_outside_and_unclipped(
    train: ModuleType,
) -> None:
    report = train.formula_unit_contract()
    assert report["pass"] is True
    assert report["maximum_formula_abs_error"] == 0.0
    assert report["maximum_inside_gaussian_abs_error"] == 0.0
    assert report["additional_clip_detected"] is False
    assert report["large_z_loss"] > 1000.0
    values = train.huber_loss(np.asarray([0.5, 2.0, 1000.0]))
    gaussian = 0.5 * np.asarray([0.5, 2.0, 1000.0]) ** 2
    assert values[0] == gaussian[0]
    assert values[1] < gaussian[1]
    assert values[2] < gaussian[2]
    assert values[2] > values[1]


def test_no_op_constant_gr_pf_is_bitwise_parent_parity(train: ModuleType) -> None:
    report = train.no_op_toy_pf_contract()
    assert report["pass"] is True
    assert report["prediction_bitwise_equal"] is True
    assert report["seed_log_likelihood_bitwise_equal"] is True
    assert report["resampling_ledger_bitwise_equal"] is True
    assert report["minimum_ess_bitwise_equal"] is True
    assert report["position_clip_ledger_bitwise_equal"] is True
    assert report["huber_linear_particle_updates"] == 0


def test_embedded_gaussian_reference_matches_exp404_kernel(
    train: ModuleType,
    exp404: ModuleType,
) -> None:
    md, z, gr, grid_gr = synthetic_pf_inputs()
    args = (
        md,
        z,
        gr,
        grid_gr,
        90.0,
        0.2,
        20.0,
        100.0,
        0.01,
        32,
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
    expected = exp404._pf_lik_allseeds(*args)
    observed = train._pf_gaussian_allseeds(*args)
    for expected_array, observed_array in zip(expected, observed, strict=True):
        np.testing.assert_array_equal(observed_array, expected_array)


def test_huber_kernel_is_deterministic_finite_and_changes_outlier_filtering(
    train: ModuleType,
) -> None:
    md, z, gr, grid_gr = synthetic_pf_inputs()
    args = (
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
        1.345,
        0.998,
        0.002,
        0.005,
        0.1,
        0.001,
        0.5,
        4.5,
    )
    first = train._pf_huber_allseeds(*args)
    second = train._pf_huber_allseeds(*args)
    for first_array, second_array in zip(first, second, strict=True):
        np.testing.assert_array_equal(first_array, second_array)
        assert np.isfinite(first_array).all()
    assert int(first[5].sum()) > 0

    gaussian = train._pf_gaussian_allseeds(
        *args[:12],
        *args[13:],
    )
    assert not np.array_equal(first[1], gaussian[1])

    extreme_args = list(args)
    extreme_args[2] = np.full_like(gr, 1.0e6)
    extreme = train._pf_huber_allseeds(*extreme_args)
    assert np.isfinite(extreme[0]).all()
    assert np.isfinite(extreme[1]).all()


def test_input_preparation_matches_exp404_x1p0(
    train: ModuleType,
    exp404: ModuleType,
) -> None:
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
    assert observed["scale_audit"]["candidate_scale"] == expected["scale_audit"]["candidate_scale"]


def test_fixed32_manifest_is_sha_pinned_and_prefreeze_reads_well_only(
    train: ModuleType,
    config: dict,
) -> None:
    observed = hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest()
    assert observed == config["data"]["fixed32_manifest"]["expected_sha256"]
    wells, report = train.load_fixed32_scope(config)
    assert len(wells) == 32
    assert len(set(wells)) == 32
    assert wells == sorted(wells)
    assert report["columns_read_before_freeze"] == ["well"]


def test_stable_seed_contract_excludes_variant_and_matches_exp404(
    train: ModuleType,
) -> None:
    wells = ["well-a", "well-b", "well-c"]
    report = train.stable_seed_contract(wells, 128)
    assert report["pass"] is True
    assert report["unique_seed_bases"] == 3
    expected = int(hashlib.sha256(b"likpf::train::well-a").hexdigest()[:16], 16) % 2_147_483_647 + 1
    assert report["rows"][0]["seed_base"] == expected
    assert report["rows"][0]["seed_last"] == expected + 127


def test_truth_late_ledger_fails_closed_until_all_wells_freeze(
    train: ModuleType,
) -> None:
    ledger = train.LeakageLedger(expected_wells=2)
    ledger.freeze("a")
    with pytest.raises(RuntimeError, match="before all fixed32 artifacts"):
        ledger.record_truth(10)
    ledger.freeze("b")
    ledger.record_control(20)
    report = ledger.report()
    assert report["all_frozen"] is True
    assert report["before_freeze"]["truth_rows"] == 10
    assert report["after_freeze"]["control_rows"] == 20


def test_prediction_and_audit_sha_readback_freeze(
    train: ModuleType,
    config: dict,
    tmp_path: Path,
) -> None:
    small = deepcopy(config)
    small["data"]["fixed32_manifest"]["expected_suffix_rows"] = 32
    results = []
    for index in range(32):
        well = f"w{index:02d}"
        frame = pd.DataFrame(
            {
                "id": [f"{well}_0"],
                "well_id": [well],
                "row_idx": np.asarray([0], dtype=np.int64),
                "suffix_offset": np.asarray([0], dtype=np.int64),
                "md_since": np.asarray([1.0], dtype=np.float64),
                "raw_gr_observed": [True],
                "well_missing_fraction": np.asarray([0.0], dtype=np.float64),
                train.PRIMARY_CANDIDATE: np.asarray([100.0 + index], dtype=np.float32),
                train.ARITHMETIC_PARITY: np.asarray([99.0 + index], dtype=np.float32),
            }
        )
        audit = {
            "well_id": well,
            "status": "ok",
            "pf_well_runs": 1,
            "seed_well_trajectories": 128,
            "particle_starts": 64000,
        }
        results.append(train.FrozenWell(well, frame, audit))
    ledger = train.LeakageLedger(expected_wells=32)
    candidate, audit, frozen = train.freeze_target_free_outputs(
        results,
        tmp_path,
        small,
        ledger,
    )
    assert len(candidate) == 32
    assert len(audit) == 32
    assert frozen["sha_readback"]["pass"] is True
    assert len(frozen["prediction_logical_sha256"]) == 64
    assert ledger.all_frozen is True


def test_stage1_all_well_gate_passes_only_from_frozen_paired_metrics(
    train: ModuleType,
    config: dict,
) -> None:
    rows = 773
    well_ids = [f"w{index:03d}" for index in range(rows)]
    frame = pd.DataFrame(
        {
            "id": [f"{well}_0" for well in well_ids],
            "well_id": well_ids,
            "row_idx": np.zeros(rows, dtype=np.int64),
            "suffix_offset": np.zeros(rows, dtype=np.int64),
            "md_since": np.where(np.arange(rows) % 2 == 0, 1200.0, 100.0),
            "raw_gr_observed": np.arange(rows) % 2 == 0,
            "well_missing_fraction": np.where(np.arange(rows) % 2 == 0, 0.1, 0.4),
            train.PRIMARY_CANDIDATE: np.zeros(rows, dtype=np.float32),
            train.ARITHMETIC_PARITY: np.zeros(rows, dtype=np.float32),
            train.PRIMARY_CONTROL: np.ones(rows, dtype=np.float32),
            "saved_exp209_hmm": np.zeros(rows, dtype=np.float64),
            "candidate_hmm_50_50": np.zeros(rows, dtype=np.float64),
            "control_hmm_50_50": np.full(rows, 0.5, dtype=np.float64),
            "true_tvt": np.zeros(rows, dtype=np.float64),
            "fold": np.arange(rows) % 5,
            "hidden_like_spatial": np.arange(rows) < 200,
            "hidden_like_typewell_purged": (np.arange(rows) >= 200)
            & (np.arange(rows) < 400),
        }
    )
    primary, by_well, blend = train.build_stage1_metric_outputs(frame)
    audit = pd.DataFrame(
        {
            "well_id": well_ids,
            "status": ["ok"] * rows,
            "pf_well_runs": np.ones(rows, dtype=np.int64),
            "seed_well_trajectories": np.full(rows, 128, dtype=np.int64),
            "particle_starts": np.full(rows, 64000, dtype=np.int64),
        }
    )
    small = deepcopy(config)
    small["validation"]["expected_rows"] = rows
    small["validation"]["expected_wells"] = rows
    small["validation"]["primary_control_rmse_ft"] = 1.0
    small["validation"]["fixed_hmm_pf_50_50_control_rmse_ft"] = 0.5
    small["data"]["expected_raw_well_identity_sha256"] = "a" * 64
    frozen = {"sha_readback": {"pass": True}}
    ledger_at_freeze = {
        "before_freeze": {
            "truth_rows": 0,
            "control_rows": 0,
            "fold_rows": 0,
            "hidden_like_rows": 0,
        }
    }
    gate = train.evaluate_stage1_gate(
        small,
        frame,
        audit,
        frozen,
        primary,
        by_well,
        blend,
        ledger_at_freeze,
        {"content_sha256": "a" * 64},
        100.0,
        1.0,
    )
    assert gate["technical_gate"]["passed"] is True
    assert gate["primary_scientific_gate"]["passed"] is True
    assert gate["fixed_exp209_hmm_pf_50_50_guard"]["passed"] is True
    assert gate["passed"] is True


def test_compact_candidate_is_not_thin_and_canonical_train_is_adopted(
    config: dict,
) -> None:
    source = SOURCE.read_text()
    for heading in (
        "Exp404 likelihood-PF input preparation",
        "Gaussian reference and fixed-Huber filtering kernels",
        "Formula, no-op PF, and stable-seed contracts",
        "Target-free candidate generation and prediction freeze",
        "Truth-late fixed32 diagnostics and technical gates",
        "Generated artifacts and Stage 0 orchestration",
        "All-well Stage 1 truth-late CV and promotion gate",
    ):
        assert heading in source
    assert "Path(__file__)" not in source
    assert "from settings import" not in source
    assert "def _pf_huber_allseeds(" in source
    assert "def run_stage0(" in source
    assert "def run_stage1(" in source
    assert COMPACT_TRAIN.exists()
    canonical = json.loads(CANONICAL_TRAIN.read_text())
    canonical_text = "\n".join("".join(cell.get("source", [])) for cell in canonical["cells"])
    assert "def _pf_huber_allseeds(" in canonical_text
    assert "def run_stage0(" in canonical_text
    assert config["execution"]["run_stage_1"] is False
    assert config["implementation"]["canonical_train_notebook_adopted"] is True
