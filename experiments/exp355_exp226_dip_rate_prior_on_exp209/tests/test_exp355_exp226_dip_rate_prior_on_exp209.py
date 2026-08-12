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
    "exp355_exp226_dip_rate_prior_on_exp209_compact_selfcontained_train.py"
)
INFERENCE_SOURCE = EXP_DIR / (
    "exp355_exp226_dip_rate_prior_on_exp209_compact_selfcontained_inference.py"
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
    return load_module(TRAIN_SOURCE, "exp355_train_test")


@pytest.fixture(scope="module")
def config():
    value = yaml.safe_load((EXP_DIR / "config.yaml").read_text())
    assert isinstance(value, dict)
    return value


def synthetic_schedule_inputs(train):
    known_rows = 30
    suffix_rows = 32
    total_rows = known_rows + suffix_rows
    md = np.arange(total_rows, dtype=float)
    z = np.zeros(total_rows, dtype=float)
    tvt_input = np.r_[100.0 + np.arange(known_rows, dtype=float), [np.nan] * suffix_rows]
    horizontal = pd.DataFrame({"MD": md, "Z": z, "TVT_input": tvt_input})

    segment_id = train.k16_segment_ids(suffix_rows, 16)
    geometry_rate = 2.0 + 0.1 * segment_id
    geop = np.empty(suffix_rows, dtype=float)
    geop[0] = 200.0
    for row in range(1, suffix_rows):
        geop[row] = geop[row - 1] + geometry_rate[row]
    geometry = pd.DataFrame(
        {
            "well_id": "well_a",
            "row_idx": np.arange(known_rows, total_rows, dtype=np.int32),
            "suffix_offset": np.arange(suffix_rows, dtype=np.int32),
            "fold": np.zeros(suffix_rows, dtype=np.int8),
            "tvt_geop": geop,
        }
    )
    return horizontal, geometry


def test_contract_is_stage0_only_zero_hmm_and_run_fail_closed(train, config):
    counts = train.validate_scientific_contract(config)
    assert counts == {
        "stage_0_variants": 1,
        "reporting_folds": 5,
        "stage_0_hmm_well_runs": 0,
        "models": 0,
        "trained_folds": 0,
        "boosters": 0,
        "parent_control_retraining": False,
        "conditional_stage_1_hmm_well_runs": 773,
    }
    with pytest.raises(RuntimeError, match="package/push/run is not approved"):
        train.validate_scientific_contract(config, require_run_approval=True)

    approved = deepcopy(config)
    approved["execution"]["run_stage_0"] = True
    train.validate_scientific_contract(approved, require_run_approval=True)

    unapproved = deepcopy(approved)
    unapproved["execution"]["kaggle_push_approved"] = False
    with pytest.raises(RuntimeError, match="package/push/run is not approved"):
        train.validate_scientific_contract(unapproved, require_run_approval=True)


def test_k16_segment_assignment_matches_exp226_row_position_edges(train):
    segment_id = train.k16_segment_ids(35, 16)
    expected = np.searchsorted(
        np.linspace(0.0, 35.0, 17)[1:],
        np.arange(1.0, 36.0),
        side="left",
    )
    np.testing.assert_array_equal(segment_id, expected)
    assert segment_id.min() == 0
    assert segment_id.max() == 15


def test_schedule_anchors_first_segment_and_ports_only_relative_rate_change(train):
    horizontal, geometry = synthetic_schedule_inputs(train)
    schedule, ledger, fallback = train.build_well_rate_schedule(
        "well_a",
        geometry,
        horizontal,
        k_segments=16,
    )
    assert fallback["parent_initial_rate"] == pytest.approx(1.0)
    assert fallback["fallback_segments"] == 0
    assert ledger.loc[0, "mu_rate"] == pytest.approx(1.0)
    assert ledger.loc[5, "geometry_delta_rate"] == pytest.approx(0.5)
    assert ledger.loc[5, "mu_rate"] == pytest.approx(1.5)
    first_segment = schedule.loc[schedule["segment_id"] == 0]
    assert np.allclose(first_segment["mu_rate"], 1.0)
    assert "tvt_true" not in schedule
    assert "tvt_pred" not in schedule
    assert "gr_delta" not in schedule


def test_invalid_first_geometry_segment_falls_back_to_parent_for_whole_well(train):
    horizontal, geometry = synthetic_schedule_inputs(train)
    rates, counts = train.segment_step_rates(
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 1.0, 2.0]),
        np.array([0, 0, 1]),
        2,
    )
    assert np.isnan(rates[0])
    assert counts[0] == 0

    broken_geometry = geometry.iloc[:1].copy()
    broken_horizontal = horizontal.iloc[:31].copy()
    broken_schedule, broken_ledger, broken_fallback = train.build_well_rate_schedule(
        "well_a",
        broken_geometry,
        broken_horizontal,
        k_segments=16,
    )
    assert not broken_fallback["first_geometry_segment_valid"]
    assert broken_ledger["geometry_fallback"].all()
    assert np.allclose(
        broken_schedule["mu_rate"],
        broken_schedule["parent_initial_rate"],
    )


def test_pre_freeze_raw_reader_excludes_truth(train, tmp_path):
    pd.DataFrame(
        {
            "MD": [0.0, 1.0],
            "Z": [0.0, -1.0],
            "TVT_input": [100.0, np.nan],
            "TVT": [100.0, 100.2],
            "error": [0.0, 99.0],
        }
    ).to_csv(tmp_path / "well_a__horizontal_well.csv", index=False)
    safe = train.load_horizontal_without_truth("well_a", tmp_path)
    assert list(safe.columns) == ["MD", "Z", "TVT_input"]


def test_truth_join_requires_unchanged_frozen_schedule(train, tmp_path):
    horizontal, geometry = synthetic_schedule_inputs(train)
    truth = horizontal["TVT_input"].copy()
    truth.loc[truth.isna()] = 130.0 + np.arange(truth.isna().sum(), dtype=float)
    raw = horizontal.copy()
    raw["TVT"] = truth
    raw.to_csv(tmp_path / "well_a__horizontal_well.csv", index=False)
    schedule, ledger, fallback = train.build_well_rate_schedule(
        "well_a",
        geometry,
        horizontal,
        k_segments=16,
    )
    frozen = train.FrozenSchedule(
        schedule=schedule,
        segment_ledger=ledger,
        fallback_summary=pd.DataFrame([fallback]),
        schedule_content_sha256=train.dataframe_content_sha256(
            schedule,
            train.SCHEDULE_CONTENT_COLUMNS,
        ),
        segment_ledger_content_sha256=train.dataframe_content_sha256(ledger),
    )
    attached = train.attach_truth_after_freeze(frozen, tmp_path)
    assert "tvt_true_readout_only" in attached
    assert np.isfinite(attached["tvt_true_readout_only"]).all()

    changed = schedule.copy()
    changed.loc[0, "mu_rate"] += 0.01
    tampered = train.FrozenSchedule(
        schedule=changed,
        segment_ledger=ledger,
        fallback_summary=pd.DataFrame([fallback]),
        schedule_content_sha256=frozen.schedule_content_sha256,
        segment_ledger_content_sha256=frozen.segment_ledger_content_sha256,
    )
    with pytest.raises(RuntimeError, match="unchanged frozen schedule"):
        train.attach_truth_after_freeze(tampered, tmp_path)


def test_stage0_gate_requires_segment_and_path_fold_improvement(train, config):
    segment_metrics = pd.DataFrame(
        [
            {
                "scope": "overall",
                "scope_value": "all",
                "baseline_rmse": 1.0,
                "candidate_rmse": 0.8,
                "gain_fraction": 0.2,
                "delta_candidate_minus_baseline": -0.2,
                "gain_baseline_minus_candidate": 0.2,
            },
            *[
                {
                    "scope": "fold",
                    "scope_value": str(fold),
                    "baseline_rmse": 1.0,
                    "candidate_rmse": 0.9,
                    "gain_fraction": 0.1,
                    "delta_candidate_minus_baseline": -0.1,
                    "gain_baseline_minus_candidate": 0.1,
                }
                for fold in range(5)
            ],
        ]
    )
    path_records = [
        {
            "scope": "overall",
            "scope_value": "all",
            "baseline_rmse": 1.0,
            "candidate_rmse": 0.8,
            "delta_candidate_minus_baseline": -0.2,
            "gain_baseline_minus_candidate": 0.2,
        },
        *[
            {
                "scope": "fold",
                "scope_value": str(fold),
                "baseline_rmse": 1.0,
                "candidate_rmse": 0.9,
                "delta_candidate_minus_baseline": -0.1,
                "gain_baseline_minus_candidate": 0.1,
            }
            for fold in range(5)
        ],
        {
            "scope": "distance",
            "scope_value": "1000_plus",
            "baseline_rmse": 1.0,
            "candidate_rmse": 0.9,
            "delta_candidate_minus_baseline": -0.1,
            "gain_baseline_minus_candidate": 0.1,
        },
        {
            "scope": "hidden_like",
            "scope_value": "verification_like_spatial",
            "baseline_rmse": 1.0,
            "candidate_rmse": 0.9,
            "delta_candidate_minus_baseline": -0.1,
            "gain_baseline_minus_candidate": 0.1,
        },
        {
            "scope": "hidden_like",
            "scope_value": "verification_like_typewell_purged",
            "baseline_rmse": 1.0,
            "candidate_rmse": 0.9,
            "delta_candidate_minus_baseline": -0.1,
            "gain_baseline_minus_candidate": 0.1,
        },
    ]
    by_well = pd.DataFrame(
        {
            "well_id": ["a", "b"],
            "delta_candidate_minus_baseline": [-0.1, 0.1],
        }
    )
    passed = train.evaluate_stage_0_gate(
        segment_metrics,
        pd.DataFrame(path_records),
        by_well,
        config,
    )
    assert passed["passed"]

    regressed = deepcopy(segment_metrics)
    fold_rows = regressed.index[regressed["scope"] == "fold"]
    regressed.loc[fold_rows[:2], "delta_candidate_minus_baseline"] = 0.01
    failed = train.evaluate_stage_0_gate(
        regressed,
        pd.DataFrame(path_records),
        by_well,
        config,
    )
    assert not failed["passed"]
    assert not failed["checks"]["segment_folds_improved"]


def test_inference_is_fail_closed_and_notebook_sources_are_self_contained(config):
    inference = load_module(INFERENCE_SOURCE, "exp355_inference_test")
    contract = inference.validate_disabled_inference(config)
    assert contract["stage_0_hmm_well_runs"] == 0
    assert contract["conditional_stage_1_hmm_well_runs"] == 773
    assert contract["canonical_train_notebook_adopted"]
    assert not contract["inference_enabled"]
    with pytest.raises(RuntimeError, match="Stage 1, inference, and submission"):
        inference.stop_disabled_inference(config)

    train_source = TRAIN_SOURCE.read_text()
    inference_source = INFERENCE_SOURCE.read_text()
    assert "__file__" not in train_source
    assert "__file__" not in inference_source
    assert "from settings import" not in train_source
    assert "from settings import" not in inference_source
