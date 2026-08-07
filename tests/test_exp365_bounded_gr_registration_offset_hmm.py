from __future__ import annotations

import copy
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "experiments" / "exp365_bounded_gr_registration_offset_hmm"
TRAIN_SOURCE = (
    EXP_DIR
    / "exp365_bounded_gr_registration_offset_hmm_compact_selfcontained_train.py"
)
INFERENCE_SOURCE = (
    EXP_DIR
    / "exp365_bounded_gr_registration_offset_hmm_compact_selfcontained_inference.py"
)


def load_module(path: Path, name: str):
    old_value = os.environ.get("EXP365_IMPORT_ONLY")
    os.environ["EXP365_IMPORT_ONLY"] = "1"
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        if old_value is None:
            os.environ.pop("EXP365_IMPORT_ONLY", None)
        else:
            os.environ["EXP365_IMPORT_ONLY"] = old_value


@pytest.fixture(scope="module")
def train():
    return load_module(TRAIN_SOURCE, "exp365_train_test")


@pytest.fixture(scope="module")
def config():
    value = yaml.safe_load((EXP_DIR / "config.yaml").read_text())
    assert isinstance(value, dict)
    return value


def test_frozen_contract_is_stage0_only_and_completed_run_is_disabled(train, config):
    contract = train.validate_scientific_contract(config)
    assert contract["offset_values_ft"] == [-6.0, -3.0, 0.0, 3.0, 6.0]
    assert contract["history_rows"] == 128
    assert contract["heldout_rows"] == 64
    assert contract["hmm_well_runs"] == 0
    assert contract["run_stage_1"] is False
    assert len(contract["contract_sha256"]) == 64
    assert config["execution"]["kaggle_push_approved"] is True
    assert config["execution"]["kaggle_execution_approved"] is True
    assert config["execution"]["run_stage_0"] is False
    assert config["execution"]["run_stage_1"] is False
    assert config["execution"]["run_inference"] is False


def test_offset_transition_has_boundary_mass_returned_to_stay(train, config):
    transition = np.asarray(
        config["model"]["gr_registration_offset"]["transition_matrix"],
        dtype=np.float64,
    )
    np.testing.assert_array_equal(transition, train.expected_offset_transition())
    np.testing.assert_allclose(transition.sum(axis=1), 1.0)
    assert transition[0, 0] == pytest.approx(511.0 / 512.0)
    assert transition[2, 2] == pytest.approx(510.0 / 512.0)
    assert np.count_nonzero(transition[0]) == 2
    assert np.count_nonzero(transition[2]) == 3


def test_rolling_windows_follow_fixed_128_64_stride64_contract(train):
    windows = train.fixed_rolling_windows(320, 128, 64, 64, 256)
    assert windows == [
        (0, 128, 192),
        (64, 192, 256),
        (128, 256, 320),
    ]
    assert train.fixed_rolling_windows(255, 128, 64, 64, 256) == []


def test_circular_control_preserves_missing_mask_and_observed_multiset(train):
    values = np.asarray([1.0, np.nan, 2.0, 3.0, np.nan, 4.0])
    shifted, offset = train.circular_shift_observed_values(values, 64)
    assert offset == 1
    np.testing.assert_array_equal(np.isnan(shifted), np.isnan(values))
    np.testing.assert_array_equal(
        np.sort(shifted[np.isfinite(shifted)]),
        np.sort(values[np.isfinite(values)]),
    )
    assert not np.array_equal(
        shifted[np.isfinite(shifted)],
        values[np.isfinite(values)],
    )


def test_probability_interval_guard_allows_only_machine_epsilon(train):
    rounded = pd.DataFrame(
        {
            "nonzero_posterior_mass": [0.0, 0.5, 1.0 + 5.0e-16],
            "boundary_posterior_mass": [-5.0e-16, 0.25, 1.0],
        }
    )
    assert train.probabilities_in_unit_interval(
        rounded,
        ["nonzero_posterior_mass", "boundary_posterior_mass"],
    )
    invalid = rounded.copy()
    invalid.loc[1, "nonzero_posterior_mass"] = 1.0 + 1.0e-8
    assert not train.probabilities_in_unit_interval(
        invalid,
        ["nonzero_posterior_mass", "boundary_posterior_mass"],
    )


def synthetic_registration_series():
    typewell_tvt = np.linspace(0.0, 260.0, 2601)
    typewell_gr = 70.0 + 15.0 * np.sin(typewell_tvt / 4.0)
    positions = np.linspace(40.0, 207.0, 168)
    observed = np.interp(positions + 3.0, typewell_tvt, typewell_gr)
    offsets = np.asarray([-6.0, -3.0, 0.0, 3.0, 6.0])
    transition = np.asarray(
        [
            [511.0 / 512.0, 1.0 / 512.0, 0.0, 0.0, 0.0],
            [1.0 / 512.0, 510.0 / 512.0, 1.0 / 512.0, 0.0, 0.0],
            [0.0, 1.0 / 512.0, 510.0 / 512.0, 1.0 / 512.0, 0.0],
            [0.0, 0.0, 1.0 / 512.0, 510.0 / 512.0, 1.0 / 512.0],
            [0.0, 0.0, 0.0, 1.0 / 512.0, 511.0 / 512.0],
        ]
    )
    initial = np.asarray([0.05, 0.15, 0.60, 0.15, 0.05])
    return typewell_tvt, typewell_gr, positions, observed, offsets, transition, initial


def test_history_filter_and_predictive_nll_recover_lookup_offset(train):
    (
        typewell_tvt,
        typewell_gr,
        positions,
        observed,
        offsets,
        transition,
        initial,
    ) = synthetic_registration_series()
    posterior = train.filter_registration_history(
        positions[:128],
        observed[:128],
        typewell_tvt,
        typewell_gr,
        offsets,
        transition,
        initial,
        sigma=2.0,
        squared_z_clip=600.0,
    )
    assert offsets[int(np.argmax(posterior))] == 3.0
    score = train.score_predictive_heldout(
        positions[128:],
        observed[128:],
        typewell_tvt,
        typewell_gr,
        offsets,
        transition,
        posterior,
        sigma=2.0,
        squared_z_clip=600.0,
    )
    assert score["observed_rows"] == 40
    assert score["model_nll"] < score["delta_zero_nll"]
    assert offsets[int(np.argmax(score["final_posterior"]))] == 3.0


def test_whole_well_stage0_reads_visible_prefix_and_emits_no_physical_prediction(
    train,
    config,
):
    (
        typewell_tvt,
        typewell_gr,
        _,
        _,
        _,
        _,
        _,
    ) = synthetic_registration_series()
    known_rows = 320
    suffix_rows = 80
    all_rows = known_rows + suffix_rows
    known_tvt = np.linspace(20.0, 220.0, known_rows)
    known_gr = np.interp(known_tvt + 3.0, typewell_tvt, typewell_gr)
    horizontal = pd.DataFrame(
        {
            "MD": np.arange(all_rows, dtype=float),
            "Z": -np.arange(all_rows, dtype=float),
            "GR": np.r_[known_gr, np.full(suffix_rows, np.nan)],
            "TVT_input": np.r_[known_tvt, np.full(suffix_rows, np.nan)],
        }
    )
    typewell = pd.DataFrame({"TVT": typewell_tvt, "GR": typewell_gr})
    ledger, posterior, manifest = train.build_well_stage0(
        "synthetic",
        horizontal,
        typewell,
        config,
    )
    assert len(ledger) == 3
    assert len(posterior) == 3
    assert manifest["known_prefix_rows"] == known_rows
    assert manifest["suffix_rows"] == suffix_rows
    assert manifest["rolling_windows"] == 3
    assert "prediction" not in " ".join(ledger.columns).lower()
    assert "true_tvt" not in ledger.columns


def synthetic_resource_manifest() -> pd.DataFrame:
    rows = []
    for index in range(30):
        suffix_rows = 512 + index * 32
        position_count = 200 + index
        rows.append(
            {
                "well_id": f"well_{index:02d}",
                "suffix_rows": suffix_rows,
                "position_grid_count": position_count,
                "rate_grid_count": 41,
                "parent_state_cell_rows": suffix_rows * position_count * 41,
            }
        )
    return pd.DataFrame(rows)


def test_resource_projection_uses_five_state_multiplier_and_extrema(
    train,
    config,
):
    selected, summary = train.build_resource_projection(
        synthetic_resource_manifest(),
        config,
    )
    assert len(selected) == 16
    assert selected["well_id"].iloc[0] == "well_00"
    assert selected["well_id"].iloc[-1] == "well_29"
    assert summary["includes_minimum_workload"]
    assert summary["includes_maximum_workload"]
    assert summary["offset_state_count"] == 5
    assert summary["projected_runtime_seconds"] == pytest.approx(
        11285.868 * 5.0
    )
    assert summary["scientific_hmm_well_runs"] == 0


def test_stage0_gate_requires_nll_control_mass_stability_and_resource(
    train,
    config,
):
    config = copy.deepcopy(config)
    config["validation"]["expected_wells"] = 5
    records = []
    posterior_records = []
    for fold in range(5):
        well_id = f"well_{fold}"
        for window_id in range(2):
            records.append(
                {
                    "well_id": well_id,
                    "window_id": window_id,
                    "fold": fold,
                    "heldout_observed_gr_rows": 64,
                    "real_model_nll": 90.0,
                    "real_delta_zero_nll": 100.0,
                    "circular_model_nll": 99.0,
                    "circular_delta_zero_nll": 100.0,
                }
            )
            posterior_records.append(
                {
                    "well_id": well_id,
                    "window_id": window_id,
                    "fold": fold,
                    "posterior_sign": 1,
                    "nonzero_posterior_mass": 0.20,
                    "boundary_posterior_mass": 0.10,
                }
            )
    ledger = pd.DataFrame(records)
    posterior = pd.DataFrame(posterior_records)
    folds = train.build_fold_metrics(ledger, posterior, config)
    manifest = synthetic_resource_manifest().iloc[:5].copy()
    manifest["well_id"] = [f"well_{fold}" for fold in range(5)]
    resource = pd.DataFrame(index=np.arange(16))
    freeze = {
        "suffix_truth_columns_read": 0,
        "physical_prediction_rows": 0,
        "exact_hmm_well_runs": 0,
        "resource_projection": {
            "includes_minimum_workload": True,
            "includes_maximum_workload": True,
            "projected_runtime_seconds": 30000.0,
            "projected_peak_rss_gb": 10.0,
        },
    }
    gate = train.evaluate_stage0_gates(
        ledger,
        posterior,
        manifest,
        resource,
        freeze,
        folds,
        config,
        debug=False,
    )
    assert gate["technical_pass"] is True
    assert gate["scientific_pass"] is True
    assert gate["stage0_pass"] is True
    freeze["resource_projection"]["projected_runtime_seconds"] = 40000.0
    gate = train.evaluate_stage0_gates(
        ledger,
        posterior,
        manifest,
        resource,
        freeze,
        folds,
        config,
        debug=False,
    )
    assert gate["scientific_pass"] is False
    assert gate["stage1_eligible"] is False


def test_inference_is_fail_closed_and_sources_are_self_contained(config):
    inference = load_module(INFERENCE_SOURCE, "exp365_inference_test")
    counts = inference.validate_disabled_inference(config)
    assert counts["offset_states"] == 5
    assert counts["exact_hmm_well_runs"] == 0
    with pytest.raises(RuntimeError, match="Stage 1 exact HMM"):
        inference.stop_disabled_inference(config)

    train_source = TRAIN_SOURCE.read_text()
    inference_source = INFERENCE_SOURCE.read_text()
    for heading in [
        "## 2. Notebook-safe runtime, configuration, path, and SHA helpers",
        "## 3. Frozen scientific and execution contract",
        "## 4. Visible-prefix input and registration-filter helpers",
        "## 5. Rolling-origin Stage 0 generation",
        "## 6. Five-state exact-HMM resource projection and SHA freeze",
        "## 7. Fold metrics and promotion gates",
        "## 8. Execution orchestration and generated artifacts",
        "## 9. Setup and configuration preview",
        "## 10. Fail-closed Stage 0 execution selection",
    ]:
        assert heading in train_source
    assert "from settings import" not in train_source
    assert "__file__" not in train_source
    assert "__file__" not in inference_source
