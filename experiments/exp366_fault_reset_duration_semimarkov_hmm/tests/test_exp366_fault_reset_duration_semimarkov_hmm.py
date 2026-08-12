from __future__ import annotations

import hashlib
import importlib.util
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = ROOT / "experiments" / "exp366_fault_reset_duration_semimarkov_hmm"
TRAIN_SOURCE = (
    EXP_DIR
    / "exp366_fault_reset_duration_semimarkov_hmm_compact_selfcontained_train.py"
)
INFERENCE_SOURCE = (
    EXP_DIR
    / "exp366_fault_reset_duration_semimarkov_hmm_compact_selfcontained_inference.py"
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
    return load_module(TRAIN_SOURCE, "exp366_train_test")


@pytest.fixture(scope="module")
def config():
    value = yaml.safe_load((EXP_DIR / "config.yaml").read_text())
    assert isinstance(value, dict)
    return value


def test_frozen_contract_is_stage0_only_and_completed_run_is_closed(train, config):
    contract = train.validate_scientific_contract(config)
    assert contract["evaluation_horizon_rows"] == 512
    assert len(contract["branches"]) == 13
    assert contract["run_stage_1"] is False
    assert contract["run_inference"] is False
    assert contract["create_submission"] is False
    assert len(contract["contract_sha256"]) == 64
    with pytest.raises(PermissionError, match="not approved"):
        train.validate_scientific_contract(config, require_run_approval=True)
    assert config["execution"]["run_stage_0"] is False
    assert config["results"]["version_2"]["stage1_eligible"] is False


def test_branch_order_is_base_then_magnitude_sign_duration(train, config):
    specs = train.fixed_branch_specs(config)
    actual = [
        (row["branch_id"], row["jump_ft"], row["duration_rows"])
        for row in specs
    ]
    assert actual == [
        ("base", 0.0, 0),
        ("jump_m6p3_h128", -6.3, 128),
        ("jump_m6p3_h256", -6.3, 256),
        ("jump_m6p3_h512", -6.3, 512),
        ("jump_p6p3_h128", 6.3, 128),
        ("jump_p6p3_h256", 6.3, 256),
        ("jump_p6p3_h512", 6.3, 512),
        ("jump_m12p6_h128", -12.6, 128),
        ("jump_m12p6_h256", -12.6, 256),
        ("jump_m12p6_h512", -12.6, 512),
        ("jump_p12p6_h128", 12.6, 128),
        ("jump_p12p6_h256", 12.6, 256),
        ("jump_p12p6_h512", 12.6, 512),
    ]


def test_reset_branch_path_holds_jump_then_returns_to_base(train):
    base = np.arange(10, dtype=float)
    path = train.reset_branch_path(base, jump_ft=6.3, duration_rows=4)
    np.testing.assert_allclose(path[:4], base[:4] + 6.3)
    np.testing.assert_array_equal(path[4:], base[4:])
    np.testing.assert_array_equal(base, np.arange(10, dtype=float))
    bounded = train.reset_branch_path(
        np.full(6, 98.0),
        jump_ft=6.3,
        duration_rows=3,
        grid_min_tvt=0.0,
        grid_max_tvt=100.0,
    )
    np.testing.assert_allclose(bounded[:3], 100.0)
    np.testing.assert_allclose(bounded[3:], 98.0)


def test_refractory_and_circular_control_are_deterministic(train):
    candidates = np.zeros(20, dtype=bool)
    candidates[[1, 3, 6, 7, 13]] = True
    accepted = train.apply_refractory(candidates, refractory_rows=5)
    assert np.flatnonzero(accepted).tolist() == [1, 6, 13]
    scores = np.where(accepted, np.arange(20, dtype=float) + 1.0, 0.0)
    first = train.circular_shift_trigger_score(scores, 512)
    second = train.circular_shift_trigger_score(scores, 512)
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, scores)


def test_robust_z_and_gaussian_emission_are_fixed(train):
    values = np.asarray([1.0, 1.0, 2.0, 3.0, 100.0])
    location, scale = train.robust_location_scale(values)
    assert location == pytest.approx(2.0)
    assert scale == pytest.approx(1.4826)
    zscore = train.robust_zscore(values, location, scale)
    assert zscore[2] == pytest.approx(0.0)
    emission = train.gaussian_log_emission(
        observed_gr=np.asarray([10.0, 20.0]),
        path_tvt=np.asarray([0.0, 1.0]),
        typewell_tvt=np.asarray([0.0, 1.0]),
        typewell_gr=np.asarray([10.0, 10.0]),
        sigma=10.0,
    )
    np.testing.assert_allclose(emission, [0.0, -0.5])


def test_raw_identity_uses_parent_logical_dataframe_sha_contract(train):
    frame = pd.DataFrame(
        {
            "well_id": ["a", "b"],
            "horizontal_raw_sha256": ["h1", "h2"],
            "typewell_raw_sha256": ["t1", "t2"],
        }
    )
    digest = hashlib.sha256()
    for column in frame.columns:
        digest.update(column.encode())
        for value in frame[column].astype(str):
            digest.update(value.encode())
            digest.update(b"\n")
    assert train.logical_dataframe_sha256(frame) == digest.hexdigest()
    assert train.logical_dataframe_sha256(frame) != train.dataframe_content_sha256(
        frame
    )


def test_ranking_tie_break_uses_frozen_branch_order(train, config):
    specs = train.fixed_branch_specs(config)
    scores = {str(row["branch_id"]): -1.0 for row in specs}
    ranking = train.ranked_branch_ids(scores, specs)
    assert ranking == [str(row["branch_id"]) for row in specs]


def test_forward_window_mse_is_aligned_to_trigger_start(train):
    squared_error = np.arange(1.0, 7.0)
    actual = train.forward_window_mse(squared_error, horizon=3)
    np.testing.assert_allclose(actual, [2.0, 3.0, 4.0, 5.0])


def test_truth_access_ledger_rejects_prefreeze_target(train):
    ledger = train.TruthAccessLedger()
    with pytest.raises(ValueError, match="forbidden pre-freeze"):
        ledger.guard_prefreeze_columns(["MD", "TVT"], 12, "synthetic")
    assert ledger.truth_rows_before_freeze == 12


def test_synthetic_prefreeze_builds_one_trigger_and_all_13_branches(
    train, config, tmp_path
):
    synthetic = deepcopy(config)
    well = "synthetic"
    prefix_rows = 20
    suffix_rows = 600
    total_rows = prefix_rows + suffix_rows
    tvt_input = np.full(total_rows, np.nan)
    tvt_input[:prefix_rows] = 0.0
    gr = np.concatenate(
        [
            np.arange(prefix_rows, dtype=float),
            np.full(suffix_rows, 100.0),
        ]
    )
    horizontal = pd.DataFrame(
        {
            "MD": np.arange(total_rows, dtype=float),
            "Z": np.zeros(total_rows),
            "GR": gr,
            "TVT_input": tvt_input,
            "TVT": np.zeros(total_rows),
        }
    )
    typewell = pd.DataFrame(
        {
            "TVT": [-100.0, 100.0],
            "GR": [0.0, 0.0],
        }
    )
    horizontal.to_csv(tmp_path / f"{well}__horizontal_well.csv", index=False)
    typewell.to_csv(tmp_path / f"{well}__typewell.csv", index=False)
    suffix_index = np.arange(prefix_rows, total_rows)
    sigma = train.exp209_prefix_sigma(
        horizontal[["MD", "Z", "GR", "TVT_input"]],
        typewell["TVT"].to_numpy(float),
        typewell["GR"].to_numpy(float),
        [10.0, 60.0],
    )
    saved = pd.DataFrame(
        {
            "id": [f"{well}_{row}" for row in suffix_index],
            "well_id": well,
            "row_idx": suffix_index,
            "hmm_mean_tvt": 0.0,
            "hmm_prefix_sigma": sigma,
        }
    )
    trigger, branches, manifest = train.build_prefreeze_rows_for_well(
        well,
        saved,
        tmp_path,
        synthetic,
        train.TruthAccessLedger(),
    )
    assert len(trigger) == suffix_rows - 512 + 1
    assert trigger["accepted_trigger"].sum() == 1
    assert len(branches) == 13
    assert branches["branch_id"].nunique() == 13
    assert branches["selected_branch"].sum() == 1
    assert manifest["accepted_trigger_rows"] == 1


def test_auc_and_event_metrics_report_positive_signal(train):
    labels = np.asarray([False, False, True, True])
    scores = np.asarray([0.0, 0.1, 0.9, 1.0])
    assert train.roc_auc_binary(labels, scores) == pytest.approx(1.0)
    events = pd.DataFrame(
        {
            "well_id": ["a", "b"],
            "alternative_within10": [True, True],
            "evidence_reciprocal_rank": [1.0, 0.5],
            "base_first_reciprocal_rank": [0.5, 0.25],
            "base_mse": [16.0, 25.0],
            "selected_mse": [4.0, 9.0],
            "oracle_mse": [1.0, 4.0],
            "selected_is_oracle": [True, False],
        }
    )
    metrics = train.event_metrics(events)
    assert metrics["alternative_branch_within10_coverage"] == pytest.approx(1.0)
    assert metrics["mrr_gain_vs_base_first"] == pytest.approx(0.375)
    assert metrics["selected_branch_rmse_gain_vs_base_ft"] > 0.0


def test_inference_is_fail_closed_and_sources_are_not_thin(config):
    inference = load_module(INFERENCE_SOURCE, "exp366_inference_test")
    counts = inference.validate_disabled_inference(config)
    assert counts["fixed_branches"] == 13
    assert counts["semimarkov_hmm_well_runs"] == 0
    with pytest.raises(RuntimeError, match="Stage 1 semi-Markov HMM"):
        inference.stop_disabled_inference(config)

    train_source = TRAIN_SOURCE.read_text()
    inference_source = INFERENCE_SOURCE.read_text()
    for heading in [
        "## 2. Notebook-safe runtime, configuration, path, and SHA helpers",
        "## 3. Frozen scientific and execution contract",
        "## 4. Saved exp209 and visible-prefix input preflight",
        "## 5. Target-free trigger and fixed reset-branch generation",
        "## 6. Reporting folds and pre-truth SHA freeze",
        "## 7. Late truth and hidden-like attachment",
        "## 8. Stage 0 metrics and promotion gates",
        "## 9. Execution orchestration and generated artifacts",
        "## 10. Setup and fail-closed execution selection",
    ]:
        assert heading in train_source
    assert "from settings import" not in train_source
    assert "__file__" not in train_source
    assert "__file__" not in inference_source
