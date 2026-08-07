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
EXP_DIR = ROOT / "experiments" / "exp322_gr_likelihood_weak_exp226_soft_shrink_readout"
TRAIN_PATH = (
    EXP_DIR / "exp322_gr_likelihood_weak_exp226_soft_shrink_readout_compact_selfcontained_train.py"
)


def load_module():
    os.environ["EXP322_IMPORT_ONLY"] = "1"
    name = "exp322_train"
    spec = importlib.util.spec_from_file_location(name, TRAIN_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


train = load_module()


@pytest.fixture
def config() -> dict:
    value = yaml.safe_load((EXP_DIR / "config.yaml").read_text())
    assert isinstance(value, dict)
    return value


def synthetic_score_inputs(
    *, suffix_rows: int = 100
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    known_rows = 30
    total_rows = known_rows + suffix_rows
    true_tvt = 100.0 + 0.2 * np.arange(total_rows, dtype=np.float64)
    horizontal = pd.DataFrame(
        {
            "MD": np.arange(total_rows, dtype=np.float64),
            "GR": 2.0 * true_tvt,
            "TVT_input": np.r_[true_tvt[:known_rows], np.full(suffix_rows, np.nan)],
        }
    )
    typewell_tvt = np.linspace(0.0, 300.0, 1201)
    typewell = pd.DataFrame({"TVT": typewell_tvt, "GR": 2.0 * typewell_tvt})
    row_idx = np.arange(known_rows, total_rows, dtype=np.int64)
    anchors = pd.DataFrame(
        {
            "well_id": "well_a",
            "fold": 0,
            "row_idx": row_idx,
            "suffix_offset": np.arange(suffix_rows, dtype=np.int64),
            "md_since_ft": np.arange(1, suffix_rows + 1, dtype=np.float64),
            "p226": true_tvt[row_idx] - 10.0,
        }
    )
    return anchors, horizontal, typewell


def test_contract_is_fixed_zero_model_and_push_is_closed_after_v2(config: dict) -> None:
    train.validate_scientific_contract(config, require_kaggle_approval=False)
    with pytest.raises(RuntimeError, match="package/push/run is not approved"):
        train.validate_scientific_contract(config, require_kaggle_approval=True)
    assert config["experiment"]["route"] == "pf_beam"
    assert config["implementation"]["enabled"] is True
    assert config["execution_contract"]["active_candidates"] == 1
    assert config["execution_contract"]["diagnostic_controls"] == 1
    assert config["execution_contract"]["lightgbm_configs"] == 0
    assert config["execution_contract"]["trained_folds"] == 0
    assert config["execution_contract"]["total_boosters"] == 0
    assert config["execution_contract"]["parent_prediction_regeneration"] is False
    assert config["execution_contract"]["kaggle_push_approved"] is False
    changed = copy.deepcopy(config)
    changed["shrink"]["alpha"] = 0.30
    with pytest.raises(ValueError, match="frozen contract changed"):
        train.validate_scientific_contract(changed, require_kaggle_approval=False)


def test_exp263_readout_fold_and_exp226_source_fold_are_audited_separately() -> None:
    audit = train.audit_saved_fold_relationship(
        ["a", "a", "b", "b", "c", "c", "d", "d", "e", "e"],
        [0, 0, 1, 1, 2, 2, 3, 3, 4, 4],
        [4, 4, 3, 3, 2, 2, 1, 1, 0, 0],
        [0, 1, 2, 3, 4],
    )
    assert audit["policy"].startswith("exp263_outer_fold_is_readout_stratum")
    assert audit["same_fold_row_fraction"] == pytest.approx(0.2)
    assert sum(audit["exp226_source_fold_counts"].values()) == 10
    assert sum(audit["exp263_readout_fold_counts"].values()) == 10


def test_saved_source_fold_must_remain_constant_within_well() -> None:
    with pytest.raises(ValueError, match="constant within each well"):
        train.audit_saved_fold_relationship(
            ["a", "a", "b", "c", "d", "e"],
            [0, 1, 1, 2, 3, 4],
            [0, 0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
        )


def test_exp263_fixed_blend_matches_virtual_cache_float_contract() -> None:
    exp226 = np.array([100.125, 12000.125], dtype=np.float32)
    likpf = np.array([104.5, 12001.5], dtype=np.float32)
    hmm = np.array([96.25, 11998.25], dtype=np.float32)
    actual = train.materialize_exp263_fixed_blend(exp226, likpf, hmm)
    expected = (
        0.50 * exp226.astype(np.float64)
        + 0.25 * likpf.astype(np.float64)
        + 0.25 * hmm.astype(np.float64)
    ).astype(np.float32)
    np.testing.assert_array_equal(actual, expected)


def test_exp280_parity_likelihood_ranks_positive_ten_shift(config: dict) -> None:
    anchors, horizontal, typewell = synthetic_score_inputs()
    scores, manifest = train.score_well_target_free(anchors, horizontal, typewell, config)
    selected = scores.loc[scores["likelihood_rank"] == 1]
    assert len(selected) == 1
    assert selected["shift_ft"].iloc[0] == 10.0
    assert manifest["blocks"] == 1
    assert manifest["score_finite_coverage"] == 1.0
    assert not train.TARGET_FREE_FORBIDDEN.intersection(scores.columns)


def test_score_path_rejects_truth_and_tie_uses_config_order(config: dict) -> None:
    anchors, horizontal, typewell = synthetic_score_inputs()
    anchors["tvt_true"] = anchors["p226"]
    with pytest.raises(ValueError, match="forbidden columns"):
        train.score_well_target_free(anchors, horizontal, typewell, config)
    np.testing.assert_array_equal(
        train.rank_descending(np.array([1.0, 1.0, 0.0])),
        np.array([1, 2, 3], dtype=np.int16),
    )


def test_block_features_compute_margin_entropy_zero_rank_and_gap(config: dict) -> None:
    shifts = np.asarray(config["likelihood"]["shift_bank_ft"], dtype=np.float64)
    likelihood = -np.abs(shifts - 2.0) / 10.0
    ranks = train.rank_descending(likelihood)
    rows = []
    for slot, shift in enumerate(shifts):
        rows.append(
            {
                "well_id": "well_a",
                "fold": 0,
                "block_id": 0,
                "block_start_suffix_offset": 0,
                "block_end_suffix_offset": 7,
                "block_start_row_idx": 10,
                "block_end_row_idx": 17,
                "block_row_count": 8,
                "md_since_min_ft": 1.0,
                "md_since_max_ft": 8.0,
                "observed_gr_share": 1.0,
                "shift_slot": slot,
                "shift_ft": shift,
                "likelihood_mean": likelihood[slot],
                "likelihood_sum": 8.0 * likelihood[slot],
                "likelihood_rank": ranks[slot],
                "native_typewell_coverage": 1.0,
                "extended_typewell_coverage": 1.0,
            }
        )
    block = train.build_block_features(pd.DataFrame(rows), config).iloc[0]
    zero_slot = int(np.flatnonzero(np.isclose(shifts, 0.0))[0])
    assert block["top1_shift_ft"] == 2.0
    assert block["zero_rank"] == ranks[zero_slot]
    assert block["best_minus_zero_gap"] == pytest.approx(likelihood.max() - likelihood[zero_slot])
    assert 0.0 <= block["normalized_entropy"] <= 1.0


def test_outer_train_threshold_excludes_evaluation_fold(config: dict) -> None:
    blocks = pd.DataFrame(
        {
            "well_id": [f"well_{fold}" for fold in range(5)],
            "fold": np.arange(5),
            "block_id": 0,
            "top1_top2_margin": [1000.0, 1.0, 2.0, 3.0, 4.0],
            "normalized_entropy": [0.0, 0.6, 0.7, 0.8, 0.9],
            "best_minus_zero_gap": [1000.0, 5.0, 6.0, 7.0, 8.0],
        }
    )
    thresholds = train.fit_outer_train_thresholds(blocks, config).set_index("fold")
    assert thresholds.loc[0, "margin_q20"] < 3.0
    assert thresholds.loc[0, "zero_gap_q20"] < 7.0
    assert thresholds.loc[1, "margin_q20"] > 2.5


def test_real_gate_and_sha_circular_control_preserve_activation_count(config: dict) -> None:
    blocks = pd.DataFrame(
        {
            "well_id": ["well_a"] * 4,
            "fold": [0] * 4,
            "block_id": np.arange(4),
            "block_start_suffix_offset": np.arange(4) * 512,
            "block_end_suffix_offset": np.arange(4) * 512 + 511,
            "block_start_row_idx": np.arange(4) * 512 + 10,
            "block_end_row_idx": np.arange(4) * 512 + 521,
            "block_row_count": [512] * 4,
            "md_since_min_ft": np.arange(4) * 512 + 1.0,
            "md_since_max_ft": np.arange(4) * 512 + 512.0,
            "observed_gr_share": [1.0, 1.0, 1.0, 0.5],
            "top1_top2_margin": [0.1, 0.1, 2.0, 0.1],
            "normalized_entropy": [0.9, 0.9, 0.9, 0.9],
            "zero_rank": [1, 10, 1, 1],
            "best_minus_zero_gap": [0.1, 2.0, 0.1, 0.1],
            "top1_shift_ft": [0.0] * 4,
        }
    )
    thresholds = pd.DataFrame(
        {
            "fold": [0],
            "outer_train_blocks": [20],
            "margin_q20": [0.5],
            "entropy_q80": [0.8],
            "zero_gap_q20": [0.5],
        }
    )
    gated = train.apply_real_and_control_gates(blocks, thresholds, config)
    np.testing.assert_array_equal(
        gated["real_block_gate"].to_numpy(bool), [True, False, False, False]
    )
    assert gated["control_block_gate"].sum() == gated["real_block_gate"].sum()
    assert gated["control_offset_blocks"].iloc[0] in {1, 2, 3}
    assert not np.array_equal(
        gated["real_block_gate"].to_numpy(bool),
        gated["control_block_gate"].to_numpy(bool),
    )
    expected_offset = train.stable_circular_offset("well_a", 4, "exp322")
    assert gated["control_offset_blocks"].iloc[0] == expected_offset


def test_bounded_shrink_keeps_near_rows_bitwise_and_clips_move(config: dict) -> None:
    base = pd.DataFrame(
        {
            "well_id": ["well_a"] * 3,
            "fold": [0] * 3,
            "row_idx": [10, 11, 12],
            "suffix_offset": [0, 1, 2],
            "md_since_ft": [249.0, 250.0, 300.0],
            "p_base": [100.0, 100.0, 100.0],
            "p226": [200.0, 200.0, 96.0],
        }
    )
    gates = pd.DataFrame(
        {
            "well_id": ["well_a"],
            "fold": [0],
            "block_id": [0],
            "real_block_gate": [True],
            "control_block_gate": [True],
        }
    )
    prediction = train.build_target_free_predictions(base, gates, config)
    np.testing.assert_array_equal(prediction["real_prediction"].to_numpy(), [100.0, 110.0, 99.0])
    assert not bool(prediction["real_eligible"].iloc[0])
    assert bool(prediction["real_eligible"].iloc[1])
    assert not train.TARGET_FREE_FORBIDDEN.intersection(prediction.columns)


def test_truth_join_requires_nonempty_frozen_contract() -> None:
    prediction = pd.DataFrame(
        {
            "well_id": ["well_a"],
            "row_idx": [10],
            "p_base": [100.0],
            "real_prediction": [100.0],
        }
    )
    truth = pd.DataFrame({"well_id": ["well_a"], "row_idx": [10], "tvt_true": [101.0]})
    with pytest.raises(ValueError, match="frozen target-free contract"):
        train.attach_truth_after_freeze(prediction, truth, target_free_contract_sha256="")
    joined = train.attach_truth_after_freeze(
        prediction, truth, target_free_contract_sha256="frozen_sha"
    )
    assert joined["tvt_true"].iloc[0] == 101.0


def test_gzip_decompressed_sha_is_primary_target_free_contract(config: dict) -> None:
    artifact = {
        "decompressed_sha256": "decompressed",
        "content_sha256": "frame",
        "schema_sha256": "schema",
    }
    manifest = pd.DataFrame({"name": ["input"], "raw_sha256": ["sha"]})
    contract = train.build_target_free_contract(config, artifact, artifact, artifact, manifest)
    assert contract["target_free_score_content_sha256"] == "decompressed"
    assert contract["target_free_score_frame_content_sha256"] == "frame"
    assert contract["target_free_score_schema_sha256"] == "schema"
    assert contract["target_free_contract_sha256"]


def test_compact_source_is_notebook_safe_and_canonical_scaffold_remains() -> None:
    source = TRAIN_PATH.read_text()
    assert "__file__" not in source
    assert "# ## Contents" in source
    assert "run_full_experiment(CONFIG)" in source
    assert (EXP_DIR / "exp322_gr_likelihood_weak_exp226_soft_shrink_readout_train.ipynb").exists()
