from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = (
    ROOT
    / "experiments"
    / "exp340_exp226_depth_alias_block_confidence_readout_on_exp264"
)
TRAIN_SOURCE = (
    EXP_DIR
    / (
        "exp340_exp226_depth_alias_block_confidence_readout_on_exp264_"
        "compact_selfcontained_train.py"
    )
)
INFERENCE_SOURCE = (
    EXP_DIR
    / (
        "exp340_exp226_depth_alias_block_confidence_readout_on_exp264_"
        "compact_selfcontained_inference.py"
    )
)


def load_module(path: Path, name: str):
    previous = os.environ.get("EXP340_IMPORT_ONLY")
    os.environ["EXP340_IMPORT_ONLY"] = "1"
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        if previous is None:
            os.environ.pop("EXP340_IMPORT_ONLY", None)
        else:
            os.environ["EXP340_IMPORT_ONLY"] = previous


@pytest.fixture(scope="module")
def train():
    return load_module(TRAIN_SOURCE, "exp340_train_test")


@pytest.fixture(scope="module")
def config():
    return yaml.safe_load((EXP_DIR / "config.yaml").read_text())


def test_stage0_contract_is_completed_and_run_approval_is_consumed(train, config):
    train.validate_scientific_contract(config)
    assert config["experiment"]["route"] == "ensemble"
    assert config["experiment"]["status"] == "stage_0_completed_guard_failed"
    assert config["implementation"]["enabled"] is True
    assert config["execution"]["implementation_approved"] is True
    assert config["execution"]["kaggle_push_approved"] is False
    assert config["execution"]["run_stage_0"] is False
    assert config["runtime"]["kaggle"]["train_run_on_push"] is False
    assert config["execution"]["kaggle_kernel_version"] == 1
    assert config["execution"]["kaggle_kernel_id_no"] == 128356047
    assert config["validation"]["fold_source"] == "exp280_exp226_group_safe_fold"
    assert (
        config["validation"]["exp264_outer_fold_policy"]
        == "provenance_only_not_required_to_equal_exp226_fold"
    )
    assert config["validation"]["truth_alignment_atol_ft"] == 0.001
    assert config["execution_contract"] == {
        "readout_families": 7,
        "controls": 1,
        "models": 0,
        "hmm_well_runs": 0,
        "trained_folds": 0,
        "boosters": 0,
        "parent_control_retraining": False,
    }
    with pytest.raises(RuntimeError, match="package/push/run is not approved"):
        train.validate_scientific_contract(config, require_run_approval=True)


def test_fixed_parent_sha_contracts_are_pinned(config):
    assert config["data"]["exp280_source"]["score_content_sha256"] == (
        "4a546cfe5f9291168bdb4dcb912182b079e0343af845f76005f6a7100ac3aa46"
    )
    assert config["data"]["exp280_source"]["score_decompressed_sha256"] == (
        "c6e9e39a5fd3944f7516e68bdd6b9d27430a47f0bfbe3c50d39f5369791d99c3"
    )
    assert config["data"]["exp264_source"]["expected_sha256"] == (
        "b11c5005ca566f76588f4e1735386c15b8f016b874701a82e1c0741c8b839ae2"
    )
    assert config["data"]["exp226_source"]["expected_decompressed_sha256"] == (
        "709eb726cc30da523f017ed0dbd0371967b88a91ddcf25578eb9356f28e4c609"
    )
    assert config["data"]["hidden_like_assignment"]["expected_sha256"] == (
        "5f9ac9fac6bb3725a7c613f09856a85bdf73b8206fd2edf1b79e8eaa9bca6597"
    )


def synthetic_scores(train) -> pd.DataFrame:
    rows = []
    shifts = train.EXPECTED_SHIFTS
    top_slots = [6, 8, 5, 10, 6, 4, 7, 2]
    for block_index, top_slot in enumerate(top_slots):
        well = "well-a" if block_index < 4 else "well-b"
        block = block_index % 4
        fold = 0 if well == "well-a" else 1
        likelihood = -np.square(np.arange(len(shifts)) - top_slot).astype(float)
        ranks = pd.Series(-likelihood).rank(method="first").astype(int).to_numpy()
        for slot, shift in enumerate(shifts):
            rows.append(
                {
                    "well_id": well,
                    "fold": fold,
                    "block_id": block,
                    "block_start_suffix_offset": block * 4,
                    "block_end_suffix_offset": block * 4 + 3,
                    "block_start_row_idx": 100 + block * 4,
                    "block_end_row_idx": 103 + block * 4,
                    "block_row_count": 4,
                    "md_since_min_ft": block * 100.0,
                    "md_since_max_ft": block * 100.0 + 30.0,
                    "md_since_mid_ft": block * 100.0 + 15.0,
                    "observed_gr_share": 1.0,
                    "shift_slot": slot,
                    "shift_ft": shift,
                    "likelihood_mean": likelihood[slot],
                    "likelihood_rank": ranks[slot],
                }
            )
    return pd.DataFrame(rows)


def test_seven_family_formulas_and_risk_orientation_are_fixed(train):
    features = train.build_target_free_block_features(synthetic_scores(train))
    assert len(features) == 8
    assert tuple(train.FAMILIES) == (
        "top1_top2_margin",
        "softmax_entropy",
        "likelihood_weighted_shift_std",
        "zero_shift_rank",
        "absolute_top1_shift",
        "top1_shift_jump_from_previous_block",
        "three_block_sign_inconsistency",
    )
    assert np.allclose(
        features["risk__top1_top2_margin"], -features["top1_top2_margin"]
    )
    for family in train.FAMILIES[1:]:
        assert np.allclose(features[f"risk__{family}"], features[family])
    first_blocks = features["block_id"].eq(0)
    assert features.loc[first_blocks, "top1_shift_jump_from_previous_block"].eq(0.0).all()
    assert features.loc[first_blocks, "three_block_sign_inconsistency"].eq(0.0).all()
    assert np.isfinite(
        features[[f"risk__{family}" for family in train.FAMILIES]].to_numpy()
    ).all()


def test_three_block_sign_inconsistency_is_pairwise_and_zero_aware(train):
    values = np.array([10.0, -5.0, 0.0, -2.0, -10.0])
    observed = train.pairwise_sign_inconsistency(values)
    assert np.allclose(observed, [0.0, 1.0, 1.0, 0.0, 0.0])


def test_circular_control_is_deterministic_nonzero_and_distribution_preserving(train):
    shift = np.array([-20.0, -5.0, 0.0, 10.0, 40.0])
    first = train.stable_nonzero_rotation("well-a", len(shift))
    second = train.stable_nonzero_rotation("well-a", len(shift))
    assert first == second
    assert 1 <= first < len(shift)
    rotated = np.roll(shift, first)
    assert np.array_equal(np.sort(rotated), np.sort(shift))
    real = train.sequence_features(shift)
    control = train.sequence_features(rotated)
    assert all(np.isfinite(values).all() for values in (*real, *control))


def test_truth_columns_fail_closed_and_ledger_requires_freeze(train):
    train.assert_no_forbidden_columns(["well_id", "fold", "likelihood_mean"])
    with pytest.raises(ValueError, match="forbidden"):
        train.assert_no_forbidden_columns(["well_id", "actual_tvt"])
    ledger = train.TruthAccessLedger()
    with pytest.raises(ValueError, match="before target-free freeze"):
        ledger.register_truth_access()
    assert ledger.count_before_freeze == 1
    clean = train.TruthAccessLedger()
    clean.mark_frozen()
    clean.register_truth_access()
    assert clean.count_before_freeze == 0


def test_weighted_block_auc_matches_expanded_row_auc(train):
    scores = np.array([0.0, 1.0, 2.0])
    positives = np.array([0, 1, 2])
    negatives = np.array([2, 1, 0])
    observed = train.weighted_block_auc(scores, positives, negatives)
    expanded_scores = np.repeat(scores, positives + negatives)
    expanded_labels = np.concatenate(
        [
            np.r_[np.ones(int(pos)), np.zeros(int(neg))]
            for pos, neg in zip(positives, negatives, strict=True)
        ]
    ).astype(bool)
    expected = train.binary_auc(expanded_labels, expanded_scores)
    assert observed == pytest.approx(expected)
    assert observed == pytest.approx(8.5 / 9.0)
    assert train.weighted_block_auc(scores, np.ones(3), np.zeros(3)) is None


def test_streaming_post_freeze_aggregators_match_blocks_without_fold_equality(
    train, tmp_path
):
    exp226_rows = []
    exp264_rows = []
    for well, first_row, exp226_fold, exp264_fold in (
        ("well-a", 10, 0, 3),
        ("well-b", 20, 1, 4),
    ):
        for suffix in range(5):
            row_idx = first_row + suffix
            truth = 100.0 + suffix
            exp226_rows.append(
                {
                    "well_id": well,
                    "row_idx": row_idx,
                    "suffix_offset": suffix,
                    "fold": exp226_fold,
                    "tvt_true": truth,
                    "tvt_pred": truth + 1.0,
                }
            )
            exp264_rows.append(
                {
                    "id": f"{well}_{row_idx}",
                    "well": well,
                    "outer_fold": exp264_fold,
                    "actual_tvt": np.float32(truth),
                    "md_since": float(suffix + 1),
                    "pred": np.float32(truth + 2.0),
                }
            )
    exp226_path = tmp_path / "exp226.csv.gz"
    exp264_path = tmp_path / "exp264.parquet"
    pd.DataFrame(exp226_rows).to_csv(
        exp226_path,
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    pd.DataFrame(exp264_rows[::-1]).sort_values(
        ["well", "id"], ascending=[False, True], kind="mergesort"
    ).to_parquet(exp264_path, index=False)
    exp226 = train.aggregate_exp226_blocks(exp226_path, block_size=4)
    exp264, evidence = train.aggregate_exp264_blocks(
        exp264_path,
        prediction_column="pred",
        block_size=4,
    )
    assert len(exp226) == len(exp264) == 4
    assert exp226["exp226_rows"].tolist() == [4, 1, 4, 1]
    assert exp264["exp264_rows"].tolist() == [4, 1, 4, 1]
    assert exp226["fold"].tolist() == [0, 0, 1, 1]
    assert exp264["exp264_outer_fold"].tolist() == [3, 3, 4, 4]
    assert evidence["rows"] == 10
    assert evidence["wells"] == 2
    assert evidence["outer_folds"] == [3, 4]
    assert evidence["rmse"] == pytest.approx(2.0)
    merged = exp226.merge(exp264, on=["well_id", "block_id"], validate="one_to_one")
    assert np.array_equal(merged["exp226_rows"], merged["exp264_rows"])
    assert np.array_equal(
        merged["exp226_first_row_idx"], merged["exp264_first_row_idx"]
    )


def passing_metric_frames(train):
    scope_rows = []
    fold_rows = []
    for family in train.FAMILIES:
        for scope in (
            "pooled",
            "distance_1000_plus",
            "hidden_like_spatial",
            "hidden_like_typewell_purged",
        ):
            scope_rows.append(
                {
                    "family": family,
                    "scope": scope,
                    "feature_finite_coverage": 1.0,
                    "q4_minus_q1_mean_exp264_block_rmse": 1.0,
                    "q4_minus_q1_median_exp264_block_rmse": 0.5,
                    "row_weighted_abs_error_ge_10ft_auc": 0.7,
                    "real_minus_circular_auc": (
                        0.1 if family in train.SEQUENCE_FAMILIES else np.nan
                    ),
                }
            )
        for fold in range(5):
            fold_rows.append(
                {
                    "family": family,
                    "fold": fold,
                    "q4_minus_q1_mean_exp264_block_rmse": 0.2,
                    "row_weighted_abs_error_ge_10ft_auc": 0.6,
                    "real_minus_circular_auc": (
                        0.05 if family in train.SEQUENCE_FAMILIES else np.nan
                    ),
                }
            )
    boundaries = pd.DataFrame(
        [
            {
                "family": family,
                "fold": fold,
                "q25_risk_boundary": 0.0,
                "q75_risk_boundary": 1.0,
            }
            for family in train.FAMILIES
            for fold in range(5)
        ]
    )
    return pd.DataFrame(scope_rows), pd.DataFrame(fold_rows), boundaries


def test_fixed_gate_requires_a_preregistered_family_to_pass_every_guard(train, config):
    scopes, folds, boundaries = passing_metric_frames(train)
    family_gate, decision = train.evaluate_fixed_gate(scopes, folds, boundaries, config)
    assert family_gate["passed"].all()
    assert decision["scientific_passed"] is True
    broken_scopes = scopes.copy()
    broken_scopes.loc[
        broken_scopes["scope"].eq("pooled"), "row_weighted_abs_error_ge_10ft_auc"
    ] = 0.59
    broken_folds = folds.copy()
    broken_folds["row_weighted_abs_error_ge_10ft_auc"] = 0.49
    broken_gate, broken_decision = train.evaluate_fixed_gate(
        broken_scopes, broken_folds, boundaries, config
    )
    assert not broken_gate["passed"].any()
    assert broken_decision["scientific_passed"] is False
    assert broken_decision["action"] == "close_depth_alias_confidence_branch_without_rescue"


def test_inference_is_fail_closed_and_sources_are_self_contained(train, config):
    inference = load_module(INFERENCE_SOURCE, "exp340_inference_test")
    contract = inference.validate_disabled_inference(config)
    assert contract["readout_families"] == 7
    assert contract["models"] == 0
    assert contract["hmm_well_runs"] == 0
    with pytest.raises(RuntimeError, match="inference, and submission are disabled"):
        inference.stop_disabled_inference(config)
    train_text = TRAIN_SOURCE.read_text()
    inference_text = INFERENCE_SOURCE.read_text()
    assert "def build_target_free_block_features" in train_text
    assert "def weighted_block_auc" in train_text
    assert "def evaluate_fixed_gate" in train_text
    assert "def run_stage_0_experiment" in train_text
    assert "from settings import" not in train_text
    assert "from src" not in train_text
    assert "Path(__file__)" not in train_text
    assert "Path(__file__)" not in inference_text
