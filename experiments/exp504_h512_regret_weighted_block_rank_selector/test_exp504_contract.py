from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


HERE = Path(__file__).parent
SOURCE = HERE / "exp504_h512_regret_weighted_block_rank_selector_compact_selfcontained_train.py"


def load_module():
    spec = importlib.util.spec_from_file_location("exp504_train", SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_config_freezes_one_rank_config_and_five_cpu_boosters() -> None:
    module = load_module()
    config = module.load_config(HERE / "config.yaml")
    execution = config["execution_contract"]
    assert execution["scientific_variants"] == 1
    assert execution["rank_configs"] == 1
    assert execution["outer_folds"] == 5
    assert execution["total_cpu_models"] == 5
    assert execution["total_boosters"] == 5
    assert execution["parent_control_retrains"] == 0
    assert execution["candidate_regeneration_runs"] == 0
    assert execution["gpu_models"] == 0
    assert config["implementation"]["inference_enabled"] is False
    assert config["implementation"]["submission_enabled"] is False


def test_block_aggregation_uses_fixed_statistics_and_all_missing_policy() -> None:
    module = load_module()
    matrix = np.array(
        [
            [1.0, np.nan],
            [3.0, np.nan],
            [5.0, np.nan],
        ],
        dtype=np.float32,
    )
    aggregated = module.aggregate_feature_matrix(matrix).reshape(2, 9)
    assert aggregated[0, 0] == pytest.approx(1.0)
    assert aggregated[0, 1] == pytest.approx(3.0)
    assert aggregated[0, 2] == pytest.approx(np.std([1.0, 3.0, 5.0], ddof=0))
    assert aggregated[0, 3] == pytest.approx(1.4)
    assert aggregated[0, 4] == pytest.approx(3.0)
    assert aggregated[0, 5] == pytest.approx(4.6)
    assert aggregated[0, 6] == pytest.approx(1.0)
    assert aggregated[0, 7] == pytest.approx(5.0)
    assert aggregated[0, 8] == pytest.approx(4.0)
    assert aggregated[1, 0] == 0.0
    assert np.isnan(aggregated[1, 1:]).all()


def test_pair_schema_is_antisymmetric_without_ordinal_candidate_index() -> None:
    module = load_module()
    row_features = ["ctx__a", "cand__x", "id__candidate__a"]
    candidate_names, shared_names, pair_names = module.block_feature_names(row_features)
    assert len(candidate_names) == 2 * 9
    assert len(shared_names) == 1 * 9
    assert len(pair_names) == 3 * 18 + 9 + 6
    assert all("candidate_index" not in name for name in pair_names)

    candidate = np.zeros((1, 2, 18), dtype=np.float32)
    candidate[0, 0] = 3.0
    candidate[0, 1] = 1.0
    shared = np.full((1, 9), 4.0, dtype=np.float32)
    context = np.arange(6, dtype=np.float32)[None, :]
    forward = module.assemble_pair_features(
        candidate, shared, context, np.array([0]), np.array([0]), np.array([1])
    )[0]
    reverse = module.assemble_pair_features(
        candidate, shared, context, np.array([0]), np.array([1]), np.array([0])
    )[0]
    assert np.allclose(forward[:18], 2.0)
    assert np.allclose(reverse[:18], -2.0)
    assert np.allclose(forward[18:36], reverse[18:36])
    assert np.allclose(forward[36:], reverse[36:])


def test_corrected_88_feature_cube_is_fully_constructible_and_ctx_is_shared() -> None:
    module = load_module()
    config = module.load_config(HERE / "config.yaml")
    features, _ = module.load_feature_schema(config)
    n_rows = 4
    n_candidates = len(module.EXPECTED_CANDIDATE_ORDER)
    base = pd.DataFrame(
        {
            "id": [f"w_{index}" for index in range(n_rows)],
            "well": ["w"] * n_rows,
            "well_row_idx": np.arange(10, 10 + n_rows, dtype=np.int32),
            "outer_fold": np.zeros(n_rows, dtype=np.int8),
            "md_since": np.arange(1, n_rows + 1, dtype=np.float32),
            "last_known_tvt": np.full(n_rows, 100.0, dtype=np.float32),
        }
    )
    values = (
        np.arange(n_rows, dtype=np.float32)[:, None]
        + np.arange(n_candidates, dtype=np.float32)[None, :]
        + 100.0
    )
    native_fields = (
        "beam_family_std",
        "candidate_finite_source",
        "geometry_gr_delta",
        "loglik_per_row",
        "score_margin",
        "selfgr_peak_tvt",
        "selfgr_quality",
        "selfgr_typewell_agreement",
        "selfgr_valid",
        "sigma_tvt",
        "source_loglik",
    )
    confidence = {}
    for candidate_id in module.PRIMITIVE_CANDIDATES:
        frame = base[module.KEY_COLUMNS].copy()
        frame["confidence_valid"] = True
        for field in native_fields:
            frame[field] = np.linspace(0.1, 0.4, n_rows, dtype=np.float32)
        confidence[candidate_id] = frame
    raw_context = pd.DataFrame(
        {
            feature: np.arange(n_rows, dtype=np.float32)
            for feature in features
            if feature.startswith("ctx__")
        }
    )
    bundle = module.FoldBundle(
        base=base,
        values=values,
        available=np.ones_like(values, dtype=bool),
        confidence=confidence,
    )
    cube = module.build_selected_feature_cube(
        bundle,
        raw_context,
        np.arange(n_rows),
        features,
        config,
    )
    assert cube.shape == (n_rows, n_candidates, 88)
    assert not np.isinf(cube).any()
    context_positions = [
        index for index, feature in enumerate(features) if feature.startswith("ctx__")
    ]
    assert np.allclose(
        cube[:, :, context_positions],
        cube[:, :1, context_positions],
        equal_nan=True,
    )
    first_id = features.index("id__candidate__exp226_k16")
    assert np.array_equal(cube[0, :, first_id], np.r_[1.0, np.zeros(11)])


def test_pair_targets_duplicate_orientations_and_preserve_unordered_weight() -> None:
    module = load_module()
    mse = np.tile(np.arange(12, dtype=np.float64), (2, 1))
    mse[1, 1] = mse[1, 0]  # Drop exactly one canonical tie.
    targets = module.build_pair_targets(
        mse,
        np.array([512, 100], dtype=np.int32),
        np.array([0, 1], dtype=np.int32),
        tie_tolerance=1.0e-12,
    )
    assert len(targets.canonical_table) == 2 * 66 - 1
    assert len(targets.block_ids) == 2 * len(targets.canonical_table)
    half = len(targets.canonical_table)
    assert np.array_equal(targets.left[:half], targets.right[half:])
    assert np.array_equal(targets.right[:half], targets.left[half:])
    assert np.array_equal(targets.label[:half], 1 - targets.label[half:])
    assert targets.canonical_table["normalized_weight"].mean() == pytest.approx(1.0)
    assert targets.sample_weight.sum() == pytest.approx(float(half))


def test_borda_tie_prefers_anchor_and_guard_rejects_nonanchor() -> None:
    module = load_module()
    neutral = np.full((1, 66), 0.5, dtype=np.float64)
    selected, provisional, fallback, borda = module.select_from_pair_probabilities(neutral)
    assert provisional[0] == module.ANCHOR_INDEX
    assert selected[0] == module.ANCHOR_INDEX
    assert not fallback[0]
    assert np.allclose(borda, 0.5)

    probabilities = np.full((1, 66), 0.5, dtype=np.float64)
    for pair_index, (left, right) in enumerate(
        zip(module.PAIR_LEFT, module.PAIR_RIGHT, strict=True)
    ):
        if left == 0:
            probabilities[0, pair_index] = 0.9
        if left == 0 and right == module.ANCHOR_INDEX:
            probabilities[0, pair_index] = 0.49
    selected, provisional, fallback, _ = module.select_from_pair_probabilities(probabilities)
    assert provisional[0] == 0
    assert selected[0] == module.ANCHOR_INDEX
    assert fallback[0]


def test_truth_ledger_blocks_outer_valid_readout_until_prediction_freeze() -> None:
    module = load_module()
    ledger = module.TruthAccessLedger()
    with pytest.raises(RuntimeError, match="target-free"):
        ledger.authorize_outer_train(0, [1, 2, 3, 4], 100)
    ledger.freeze_target_free()
    with pytest.raises(RuntimeError, match="outer-valid"):
        ledger.authorize_outer_train(0, [0, 1, 2, 3], 100)
    ledger.authorize_outer_train(0, [1, 2, 3, 4], 100)
    with pytest.raises(RuntimeError, match="prediction freeze"):
        ledger.authorize_outer_valid(0, 20)
    ledger.freeze_prediction(0, "a" * 64)
    ledger.authorize_outer_valid(0, 20)


def test_promotion_gate_is_all_and() -> None:
    module = load_module()
    config = module.load_config(HERE / "config.yaml")
    scope = pd.DataFrame(
        {
            "scope": [
                "pooled",
                "md_since_0_250",
                "md_since_250_1000",
                "md_since_1000_plus",
                "hidden_like_spatial",
                "hidden_like_typewell_purged",
            ],
            "selected_rmse_ft": [8.18] * 6,
            "anchor_rmse_ft": [8.24] * 6,
            "delta_vs_anchor_ft": [-0.06, 0.0, 0.01, 0.02, -0.01, 0.0],
        }
    )
    folds = pd.DataFrame({"delta_vs_anchor_ft": [-0.1, -0.1, -0.1, -0.1, 0.01]})
    by_well = pd.DataFrame({"delta_vs_anchor_ft": [-0.1, 0.0, 0.1, 0.2]})
    passed = module.evaluate_promotion_gates(scope, folds, by_well, {"x": True}, config)
    assert passed["passed"]
    by_well.loc[3, "delta_vs_anchor_ft"] = 0.251
    failed = module.evaluate_promotion_gates(scope, folds, by_well, {"x": True}, config)
    assert not failed["passed"]
    assert failed["checks"]["worst_well_delta_within_0p25_ft"] is False


def test_source_contains_no_candidate_regeneration_or_inference_flow() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    forbidden_calls = (
        "run_particle_filter(",
        "run_pf(",
        "run_beam(",
        "run_hmm(",
        "submission.csv",
        "LGBMRegressor(",
    )
    assert all(call not in source for call in forbidden_calls)
    assert "Path(__file__)" not in source
