from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
SOURCE = (
    HERE
    / "exp507_exp504_nested_rank_compact_addonly_on_exp413_compact_selfcontained_train.py"
)


def load_module():
    root = HERE.parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    os.environ["EXP507_IMPORT_ONLY"] = "1"
    spec = importlib.util.spec_from_file_location("exp507_train", SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_frozen_schema_and_compute_contract() -> None:
    module = load_module()
    assert len(module.CANDIDATE_ORDER) == 12
    assert len(module.PAIR_LEFT) == 66
    assert len(module.RANK_COMPACT_FEATURES) == 45
    assert len(set(module.RANK_COMPACT_FEATURES)) == 45
    assert len(module.BLOCK_CONSTANT_FEATURES) == 42
    assert module.ROW_VARYING_FEATURES == [
        "rank_borda_tvt_mean",
        "rank_borda_tvt_std",
        "rank_h512_relative_position",
    ]
    assert module.STATIC_CONTRACT["cost_contract"] == {
        "rank_models": 20,
        "tvt_models": 15,
        "total": 35,
        "control_retrains": 0,
        "outer_retrains": 0,
    }
    assert module.CONFIG["implementation"]["kaggle_stage_n_approval_received"] is True
    assert module.CONFIG["execution_contract"]["stage_n"]["status"] == "complete_technical_pass"
    assert module.CONFIG["data"]["stage_n_source"]["expected_manifest_sha256"] == (
        "9a126024f0a67ab571e053038aa4a36e8b6773b6f0ff839d1fdf9ec63bcb7735"
    )
    assert module.CONFIG["runtime"]["stage_n_run_approved"] is False
    assert module.CONFIG["runtime"]["stage_d_run_approved"] is False
    assert module.CONFIG["implementation"]["kaggle_stage_d_approval_received"] is True
    assert module.CONFIG["implementation"]["stage_d_executed"] is True
    assert module.CONFIG["execution_contract"]["stage_d"]["status"] == (
        "complete_technical_pass_scientific_fail_closed"
    )
    assert module.CONFIG["implementation"]["inference_enabled"] is False
    assert module.CONFIG["implementation"]["submission_enabled"] is False


def test_all_ties_put_anchor_first_without_fallback() -> None:
    module = load_module()
    probability = np.full((2, 66), 0.5, dtype=np.float32)
    rank = module.pair_probability_to_rank(probability)
    assert np.allclose(rank["borda"], 0.5)
    assert np.allclose(rank["borda"].sum(axis=1), 6.0)
    assert np.array_equal(rank["provisional"], [module.ANCHOR_INDEX] * 2)
    assert np.array_equal(rank["order"][:, 0], [module.ANCHOR_INDEX] * 2)
    assert not rank["fallback"].any()
    assert np.allclose(rank["weights"], 1.0 / 12.0)
    assert np.allclose(rank["entropy"], 1.0)


def test_anchor_guard_falls_back_at_exactly_half() -> None:
    module = load_module()
    probability = np.full((1, 66), 0.5, dtype=np.float32)
    for index, (left, right) in enumerate(
        zip(module.PAIR_LEFT, module.PAIR_RIGHT, strict=True)
    ):
        if left == 0 and right != module.ANCHOR_INDEX:
            probability[0, index] = 0.9
    rank = module.pair_probability_to_rank(probability)
    assert rank["provisional"][0] == 0
    assert bool(rank["fallback"][0]) is True
    anchor_pair_position = np.flatnonzero(
        (module.PAIR_LEFT == 0) & (module.PAIR_RIGHT == module.ANCHOR_INDEX)
    )[0]
    assert probability[0, anchor_pair_position] == 0.5


def test_compact_builder_has_exact_rowwise_moments_and_relative_position() -> None:
    module = load_module()
    rows = pd.DataFrame(
        {
            "id": ["a0", "a1", "a2", "b0"],
            "well": ["a", "a", "a", "b"],
            "well_row_idx": [0, 1, 2, 0],
            "outer_fold": [0, 0, 0, 1],
            "md_since": [0.0, 1.0, 2.0, 0.0],
            "h512_group": [0, 0, 0, 1],
        }
    )
    blocks = pd.DataFrame(
        {
            "h512_group": [0, 1],
            "well": ["a", "b"],
            "outer_fold": [0, 1],
            "row_start": [0, 3],
            "row_stop_exclusive": [3, 4],
            "row_count": [3, 1],
        }
    )
    values = np.arange(48, dtype=np.float32).reshape(4, 12)
    surface = module.FrozenRankSurface(
        row_metadata=rows,
        blocks=blocks,
        candidate_values=values,
        candidate_features=np.empty((2, 12, 0), dtype=np.float32),
        shared_features=np.empty((2, 0), dtype=np.float32),
        block_context=np.empty((2, 0), dtype=np.float32),
        pair_feature_names=[],
        input_evidence={},
    )
    probability = np.full((2, 66), 0.5, dtype=np.float32)
    compact = module.build_rank_compact_partition(
        surface=surface,
        block_ids=np.array([0, 1], dtype=np.int32),
        pair_probability=probability,
        downstream_outer_fold=2,
        role="train",
        held_inner_fold=0,
    )
    assert list(compact.columns[-45:]) == module.RANK_COMPACT_FEATURES
    assert compact["rank_h512_relative_position"].tolist() == [0.0, 0.5, 1.0, 0.0]
    assert np.allclose(compact["rank_borda_tvt_mean"], values.mean(axis=1))
    assert np.allclose(compact["rank_borda_tvt_std"], values.std(axis=1, ddof=0))
    assert compact[module.RANK_COMPACT_FEATURES].to_numpy().dtype == np.float32
    assert compact["rank_anchor_rank"].eq(1.0).all()
    assert compact[f"rank_provisional_is__{module.ANCHOR_CANDIDATE}"].eq(1.0).all()


def test_pair_target_weight_normalization_and_bidirectional_labels() -> None:
    module = load_module()
    mse = np.full((1, 12), np.nan, dtype=np.float64)
    mse[0] = np.arange(12, dtype=np.float64)
    targets = module.build_pair_targets(
        mse, np.array([512], dtype=np.int32), np.array([0], dtype=np.int32)
    )
    assert targets.unordered_examples == 66
    assert len(targets.label) == 132
    assert np.array_equal(targets.label[:66], 1 - targets.label[66:])
    assert np.isclose(targets.sample_weight.sum(), 66.0)
    assert len(targets.logical_sha256) == 64


def test_nested_plan_excludes_held_outer_and_inner_and_reuses_outer_valid() -> None:
    module = load_module()
    plan = module.nested_partition_plan()
    assert len(plan) == 25
    assert sum(item["new_rank_models"] for item in plan) == 20
    for outer_fold in range(5):
        selected = [item for item in plan if item["downstream_outer_fold"] == outer_fold]
        assert len([item for item in selected if item["role"] == "train"]) == 4
        valid = [item for item in selected if item["role"] == "valid"]
        assert len(valid) == 1
        assert valid[0]["source_outer_fold"] == outer_fold
        assert valid[0]["new_rank_models"] == 0
        for item in selected:
            assert outer_fold not in item["rank_training_folds"]
            if item["role"] == "train":
                assert item["held_inner_fold"] not in item["rank_training_folds"]
                assert len(item["rank_training_folds"]) == 3
            else:
                assert len(item["rank_training_folds"]) == 4


def test_source_has_no_inference_submission_or_forbidden_feature_path() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    assert "submission.csv" not in source
    assert "run_particle_filter(" not in source
    assert "run_beam(" not in source
    module = load_module()
    forbidden = ("selected", "candidate_index", "true_tvt", "error", "oracle", "well_id")
    assert not [
        feature
        for feature in module.RANK_COMPACT_FEATURES
        if any(token in feature for token in forbidden)
    ]
