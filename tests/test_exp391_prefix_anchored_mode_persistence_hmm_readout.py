from __future__ import annotations

import os
import runpy
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "experiments" / (
    "exp391_prefix_anchored_mode_persistence_hmm_readout"
)
TRAIN = EXP_DIR / (
    "exp391_prefix_anchored_mode_persistence_hmm_readout_"
    "compact_selfcontained_train.py"
)
INFERENCE = EXP_DIR / (
    "exp391_prefix_anchored_mode_persistence_hmm_readout_"
    "compact_selfcontained_inference.py"
)
CONFIG = EXP_DIR / "config.yaml"
PARENT_HMM = ROOT / "experiments" / (
    "exp209_exp072_exp205_joint_exact_parity_fast_cache_generation"
) / "exact_hmm_smoother.py"
EXP270_TRAIN = ROOT / "experiments" / (
    "exp270_exact_hmm_posterior_mode_candidate_audit"
) / "exp270_exact_hmm_posterior_mode_candidate_audit_train.py"


def load_namespace(path: Path = TRAIN) -> dict[str, object]:
    previous = os.environ.get("EXP391_IMPORT_ONLY")
    os.environ["EXP391_IMPORT_ONLY"] = "1"
    try:
        return runpy.run_path(str(path))
    finally:
        if previous is None:
            os.environ.pop("EXP391_IMPORT_ONLY", None)
        else:
            os.environ["EXP391_IMPORT_ONLY"] = previous


@pytest.fixture(scope="module")
def module() -> dict[str, object]:
    return load_namespace()


@pytest.fixture()
def config() -> dict:
    value = yaml.safe_load(CONFIG.read_text())
    assert isinstance(value, dict)
    return value


def test_stage_a1_completed_fail_closed_and_stage_b_remains_disabled(
    module: dict[str, object],
    config: dict,
) -> None:
    module["validate_execution_contract"](
        config,
        require_run_authorization=False,
    )
    assert config["experiment"]["route"] == "pf_beam"
    assert config["experiment"]["status"] == "completed_stage_a1_fail_closed"
    assert config["implementation"]["enabled"] is True
    assert config["execution"]["implementation_approved"] is True
    assert (
        config["execution"]["implementation_approval_source"]
        == "user_message_implement_exp391_2026_07_25"
    )
    assert (
        config["execution"]["kaggle_execution_approval_source"]
        == "user_message_execute_exp391_2026_07_25"
    )
    assert (
        config["execution"]["stage_a1_execution_approval_source"]
        == "user_message_advance_to_stage_a1_2026_07_25"
    )
    assert config["execution"]["run_stage"] == "none"
    assert config["execution"]["canonical_train_notebook_adopted"] is True
    assert config["execution"]["kaggle_package_approved"] is True
    assert config["execution"]["kaggle_push_approved"] is True
    assert config["execution"]["stage_a0_run_approved"] is True
    assert config["execution"]["stage_a1_run_approved"] is True
    assert config["execution"]["stage_b_run_approved"] is False
    assert config["stage_b"]["enabled"] is False
    assert config["execution"]["run_hmm"] is False
    assert config["execution"]["inference_approved"] is False
    assert config["execution"]["submission_approved"] is False
    with pytest.raises(RuntimeError, match="no authorized run stage"):
        module["validate_execution_contract"](
            config,
            require_run_authorization=True,
        )


def test_role_read_ledger_enforces_truth_late_boundary(
    module: dict[str, object],
) -> None:
    ledger = module["RoleReadLedger"]()
    ledger.record_target_free("safe", ["well_id", "row_idx"], 2, ["target"])
    with pytest.raises(ValueError, match="forbidden pre-freeze"):
        ledger.record_target_free("bad", ["well_id", "target"], 2, ["target"])
    fresh = module["RoleReadLedger"]()
    with pytest.raises(RuntimeError, match="before target-free SHA freeze"):
        fresh.record_truth_late("truth", 2)
    ready = module["RoleReadLedger"]()
    ready.freeze()
    ready.record_truth_late("truth", 2)
    ready.record_hidden_late("role", 1)
    assert ready.truth_rows_after_freeze == 2
    assert ready.hidden_role_rows_after_freeze == 1


def synthetic_join_inputs() -> tuple[pd.DataFrame, ...]:
    keys = pd.DataFrame(
        {
            "well_id": ["w0", "w0", "w1", "w1"],
            "row_idx": [4, 5, 4, 5],
        }
    )
    exp270 = keys.assign(
        id=["w0_4", "w0_5", "w1_4", "w1_5"],
        last_known_tvt=100.0,
        md_since=[1.0, 2.0, 1.0, 2.0],
        prefix_rows=4,
        posterior_mean=[100.0, 100.1, 200.0, 200.1],
        marginal_map=[100.0, 106.1, 200.0, 206.1],
        global_viterbi=[100.0, 106.2, 200.0, 206.2],
    )
    exp209 = keys.assign(
        exp209_saved_posterior_mean=[100.0, 100.1, 200.0, 200.1]
    )
    exp226 = keys.assign(
        suffix_offset=[0, 1, 0, 1],
        fold=[0, 0, 1, 1],
        k16_preprojection=[100.0, 101.0, 200.0, 201.0],
        k16_postprojection=[100.0, 101.5, 200.0, 201.5],
    )
    exp263 = keys.assign(
        exp263_id=["w0_4", "w0_5", "w1_4", "w1_5"],
        exp263_md_since=[1.0, 2.0, 1.0, 2.0],
        exp263_partition_fold=[4, 4, 3, 3],
        exp226_k16=[100.0, 101.5, 200.0, 201.5],
        likpf_mean=[99.0, 100.0, 199.0, 200.0],
        exact_hmm=[100.0, 100.1, 200.0, 200.1],
        exp263_fixed_candidate=[99.75, 100.775, 199.75, 200.775],
    )
    return exp270, exp209, exp226, exp263


def test_strict_join_uses_exp226_reporting_fold_and_exp263_row_identity(
    module: dict[str, object],
) -> None:
    frames = synthetic_join_inputs()
    joined, summary = module["strict_target_free_join"](*frames)
    assert len(joined) == 4
    assert summary["duplicate_keys"] == 0
    assert summary["exp270_exp209_mean_max_abs_diff_ft"] == pytest.approx(0.0)
    assert summary["fold_mismatches"] == 0
    assert summary["exp263_cache_partition_folds"] == [3, 4]
    assert summary["exp226_exp263_fold_label_agreement_fraction"] == pytest.approx(0.0)
    assert joined["fold"].tolist() == [0, 0, 1, 1]
    broken = frames[-1].copy()
    broken.loc[0, "exp263_id"] = "wrong"
    with pytest.raises(ValueError, match="id mismatch"):
        module["strict_target_free_join"](*frames[:-1], broken)
    broken = frames[-1].copy()
    broken.loc[0, "exp263_md_since"] = 999.0
    with pytest.raises(ValueError, match="md_since mismatch"):
        module["strict_target_free_join"](*frames[:-1], broken)


def test_event_extraction_requires_persistent_gap_and_merges_short_break(
    module: dict[str, object],
) -> None:
    rows = 90
    mean = np.zeros(rows)
    map_path = np.zeros(rows)
    map_path[0:35] = 7.0
    map_path[40:80] = 7.0
    frame = pd.DataFrame(
        {
            "well_id": "w",
            "row_idx": np.arange(100, 100 + rows),
            "fold": 0,
            "suffix_offset": np.arange(rows),
            "posterior_mean": mean,
            "marginal_map": map_path,
            "global_viterbi": mean,
            "k16_preprojection": np.linspace(0.0, 1.0, rows),
            "k16_postprojection": np.linspace(0.0, 2.0, rows),
            "exp263_fixed_candidate": np.linspace(0.0, 3.0, rows),
        }
    )
    events = module["extract_decoder_separation_events"](
        frame,
        minimum_gap_ft=6.0,
        minimum_rows=32,
        merge_gap_rows=32,
    )
    assert len(events) == 1
    assert events.iloc[0]["start_row_idx"] == 100
    assert events.iloc[0]["end_row_idx"] == 179
    assert events.iloc[0]["rows"] == 80


def test_preflight_selection_is_target_free_fold_covered_and_stable(
    module: dict[str, object],
) -> None:
    rows = [
        {"fold": fold, "well_id": f"w{fold}_{index}", "severity": 100 - 10 * fold - index}
        for fold in range(5)
        for index in range(5)
    ]
    severity = pd.DataFrame(rows)
    first = module["select_preflight_wells"](
        severity,
        expected_folds=[0, 1, 2, 3, 4],
        per_fold=3,
        total_wells=16,
    )
    second = module["select_preflight_wells"](
        severity.sample(frac=1.0, random_state=9),
        expected_folds=[0, 1, 2, 3, 4],
        per_fold=3,
        total_wells=16,
    )
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 16
    assert set(first["fold"]) == {0, 1, 2, 3, 4}
    assert not any("target" in column or "error" in column for column in first.columns)


def bimodal_row(
    left_mass: float,
    right_mass: float,
) -> tuple[np.ndarray, np.ndarray]:
    grid = np.arange(11, dtype=np.float64)
    values = np.zeros(11, dtype=np.float64)
    values[2] = left_mass
    values[3] = left_mass * 0.05
    values[7] = right_mass * 0.05
    values[8] = right_mass
    values += 1.0e-6
    values /= values.sum()
    return values, grid


def test_mass_rank_swap_does_not_change_transition_overlap_identity(
    module: dict[str, object],
    config: dict,
) -> None:
    posterior_config = deepcopy(config["model"]["posterior_modes"])
    posterior_config["min_peak_height"] = 0.01
    first_probs, grid = bimodal_row(0.60, 0.40)
    second_probs, _ = bimodal_row(0.40, 0.60)
    first = module["extract_row_basins"](
        first_probs,
        grid,
        posterior_config,
        row_index=0,
    )
    second = module["extract_row_basins"](
        second_probs,
        grid,
        posterior_config,
        row_index=1,
    )
    assert first[0].display_rank == 1
    assert second[0].display_rank == 2
    tracked, anchor, _ = module["track_mode_lineages"](
        [first, second],
        [np.asarray([[0.39, 0.01], [0.01, 0.39]])],
        anchor_tvt=2.0,
        allowance_ft=6.0,
    )
    assert tracked[0][0].mode_id == tracked[1][0].mode_id == anchor
    assert tracked[0][1].mode_id == tracked[1][1].mode_id


def test_matching_tie_break_is_previous_id_then_current_center(
    module: dict[str, object],
) -> None:
    matching = module["maximum_weight_matching"](
        np.ones((2, 2)),
        ["mode_000", "mode_001"],
        [2.0, 8.0],
        [2.0, 8.0],
        10.0,
    )
    assert [(left, right) for left, right, _ in matching] == [(0, 0), (1, 1)]


def test_anchor_mode_uses_transported_start_prior_overlap(
    module: dict[str, object],
) -> None:
    posterior_config = {
        "min_peak_height": 0.01,
        "min_top2_mass": 0.10,
        "min_top2_to_top1_mass_ratio": 0.25,
        "min_peak_separation_ft": 4.0,
        "min_valley_depth": 0.30,
    }
    probabilities, grid = bimodal_row(0.55, 0.45)
    basins = module["extract_row_basins"](
        probabilities,
        grid,
        posterior_config,
        row_index=0,
    )
    tracked, anchor, _ = module["track_mode_lineages"](
        [basins],
        [],
        anchor_tvt=2.0,
        allowance_ft=6.0,
        anchor_overlaps=[0.1, 0.9],
    )
    assert tracked[0][0].basin.center_tvt == pytest.approx(2.0)
    assert anchor == tracked[0][1].mode_id


def tracked_basin(
    module: dict[str, object],
    *,
    row: int,
    index: int,
    left: int,
    right: int,
    center: float,
    mode_id: str,
    status: str = "matched",
) -> object:
    basin = module["Basin"](
        row_index=row,
        basin_index=index,
        left_index=left,
        right_index=right,
        peak_index=(left + right) // 2,
        center_tvt=center,
        peak_density=0.4,
        mass=0.5,
        conditional_mean=center,
        display_rank=index + 1,
        eligible_bimodal=True,
    )
    return module["TrackedBasin"](
        basin=basin,
        mode_id=mode_id,
        parent_mode_id=mode_id,
        lineage_status=status,
        transported_overlap=0.4,
    )


def test_gradual_small_steps_still_increment_cross_mode_switch_count(
    module: dict[str, object],
) -> None:
    rows = []
    for row in range(5):
        boundary = 4 + row
        rows.append(
            [
                tracked_basin(
                    module,
                    row=row,
                    index=0,
                    left=0,
                    right=boundary,
                    center=float(boundary - 2),
                    mode_id="mode_000",
                ),
                tracked_basin(
                    module,
                    row=row,
                    index=1,
                    left=boundary + 1,
                    right=12,
                    center=float(boundary + 3),
                    mode_id="mode_001",
                ),
            ]
        )
    # Every position step is only two grid cells, but the path enters mode_001.
    path = [2, 4, 6, 8, 10]
    ledger = module["annotate_path_switches"](path, rows, "mode_000")
    assert np.max(np.abs(np.diff(path))) == 2
    assert ledger["mode_switch_count"].iloc[-1] >= 1
    assert ledger["cross_mode_edge"].any()


def test_merge_split_are_ledgered_and_anchor_mask_fails_only_if_unresolved(
    module: dict[str, object],
) -> None:
    posterior_config = {
        "min_peak_height": 0.01,
        "min_top2_mass": 0.10,
        "min_top2_to_top1_mass_ratio": 0.25,
        "min_peak_separation_ft": 4.0,
        "min_valley_depth": 0.30,
    }
    two_probs, grid = bimodal_row(0.55, 0.45)
    one_probs = np.zeros_like(two_probs)
    one_probs[4] = 1.0
    rows = [
        module["extract_row_basins"](two_probs, grid, posterior_config, row_index=0),
        module["extract_row_basins"](one_probs, grid, posterior_config, row_index=1),
        module["extract_row_basins"](two_probs, grid, posterior_config, row_index=2),
    ]
    tracked, anchor, ancestry = module["track_mode_lineages"](
        rows,
        [
            np.asarray([[0.40], [0.10]]),
            np.asarray([[0.35, 0.15]]),
        ],
        anchor_tvt=2.0,
        allowance_ft=10.0,
    )
    statuses = {row["lineage_status"] for row in ancestry}
    assert "merge_survivor" in statuses
    assert "split_matched" in statuses
    assert "split_new" in statuses
    mask, unresolved = module["anchor_position_mask"](tracked, anchor, len(grid))
    assert not unresolved
    assert np.all(mask.sum(axis=1) > 0)


def test_exact_joint_kernel_preserves_exp209_marginal_parity(
    module: dict[str, object],
) -> None:
    parent = runpy.run_path(str(PARENT_HMM))
    emission = np.asarray(
        [[-1.2, -0.1, -2.7], [-2.0, -0.4, -1.1]],
        dtype=np.float32,
    )
    allowed = np.ones_like(emission, dtype=np.uint8)
    args = (
        emission,
        np.asarray([1.0, 1.3], dtype=np.float64),
        np.asarray([0.08, -0.03], dtype=np.float64),
        0.35,
        np.asarray([-0.08, 0.0, 0.08], dtype=np.float64),
        0.02,
        0.12,
        1.1,
        0.75,
        0.01,
        0.08,
        1.0,
        0.998,
    )
    actual, joint, actual_log_likelihood = module["_hmm2_fb_joint"](
        emission,
        allowed,
        *args[1:],
    )
    expected, expected_log_likelihood = parent["_hmm2_fb"](*args)
    np.testing.assert_allclose(actual, expected, atol=1.0e-8)
    np.testing.assert_allclose(joint.sum(axis=2), actual, atol=1.0e-7)
    np.testing.assert_allclose(joint.sum(axis=(1, 2)), 1.0, atol=1.0e-7)
    assert actual_log_likelihood == pytest.approx(expected_log_likelihood, abs=1.0e-6)


def test_exact_viterbi_kernel_matches_exp270_global_top1(
    module: dict[str, object],
) -> None:
    previous = os.environ.get("EXP270_IMPORT_ONLY")
    os.environ["EXP270_IMPORT_ONLY"] = "1"
    try:
        exp270 = runpy.run_path(str(EXP270_TRAIN))
    finally:
        if previous is None:
            os.environ.pop("EXP270_IMPORT_ONLY", None)
        else:
            os.environ["EXP270_IMPORT_ONLY"] = previous
    emission = np.asarray(
        [[-1.2, -0.1, -2.7], [-2.0, -0.4, -1.1], [-0.2, -1.0, -2.0]],
        dtype=np.float32,
    )
    args = (
        emission,
        np.asarray([1.0, 1.3, 0.9], dtype=np.float64),
        np.asarray([0.08, -0.03, 0.02], dtype=np.float64),
        0.35,
        np.asarray([-0.08, 0.0, 0.08], dtype=np.float64),
        0.02,
        0.12,
        1.1,
        0.75,
        0.01,
        0.08,
        1.0,
        0.998,
    )
    score, positions, rates = module["_hmm2_viterbi"](*args)
    expected_scores, expected_positions, expected_rates = exp270["_hmm2_topk"](
        *args,
        1,
    )
    assert float(score) == pytest.approx(float(expected_scores[0]), abs=1.0e-6)
    np.testing.assert_array_equal(positions, expected_positions[0])
    np.testing.assert_array_equal(rates, expected_rates[0])


def test_logical_sha_is_row_order_sensitive_and_repeatable(
    module: dict[str, object],
) -> None:
    frame = pd.DataFrame({"well_id": ["a", "b"], "row_idx": [1, 2], "value": [3.0, 4.0]})
    first = module["logical_frame_sha256"](frame)
    second = module["logical_frame_sha256"](frame.copy())
    reversed_sha = module["logical_frame_sha256"](frame.iloc[::-1])
    assert first == second
    assert first != reversed_sha


def test_selfcontained_sources_and_fail_closed_inference_contract(
    module: dict[str, object],
    config: dict,
) -> None:
    train_source = TRAIN.read_text()
    inference_source = INFERENCE.read_text()
    assert "from settings import" not in train_source
    assert "__file__" not in train_source
    assert "# ## Contents" in train_source
    assert "exp236_row_artifact_reads" in train_source
    assert "pd.read_csv(exp236" not in train_source
    assert "current_test" not in module["decoder_contract_manifest"](config)
    inference = load_namespace(INFERENCE)
    inference["validate_inference_disabled"](config)
    blockers = inference["inference_blockers"](config)
    assert any("train-side" in blocker for blocker in blockers)
    assert "RuntimeError" in inference_source
