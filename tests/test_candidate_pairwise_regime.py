from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.candidate_pairwise_regime import (
    PrimitiveFold,
    assert_target_free_feature_schema,
    build_block_fingerprints,
    evaluate_regime_separability,
    evaluate_regime_separability_from_parquet,
    feature_columns,
    fit_outer_fold_regimes,
    pair_ids,
    primitive_ids,
    validate_regime_contract,
)

PRIMITIVES = [
    "exp226_k16",
    "selfgr_hmm_a070",
    "likpf_mean",
    "exact_hmm",
    "pf_ancc",
    "beam_mean",
]


def _contract() -> dict:
    return {
        "primitives": [
            {"id": candidate_id, "family": f"family_{index}"}
            for index, candidate_id in enumerate(PRIMITIVES)
        ],
        "pair_policy": {"expected_count": 15},
        "feature_policy": {
            "candidate_absolute_value": "forbidden",
            "target_or_error_derived": "forbidden",
        },
    }


def _feature_config(block_size: int = 4) -> dict:
    return {
        "features": {
            "block_size": block_size,
            "raw_context": {
                "horizontal_numeric_allowlist": ["MD", "X", "GR"],
                "forbidden_columns": ["TVT", "TVT_input", "target", "true_tvt"],
            },
            "confidence_slots": ["sigma_tvt", "entropy", "support_count"],
            "forbidden_feature_tokens": [
                "tvt",
                "tvt_input",
                "true_tvt",
                "candidate_tvt",
                "last_known_tvt",
                "target",
                "truth",
                "error",
                "actual_abs_error",
                "pred_abs_error",
                "oracle",
                "label",
                "last_known",
            ],
        },
        "guards": {"technical": {"expected_pairs": 15}},
    }


def _bundle(shared_translation: np.ndarray | None = None) -> PrimitiveFold:
    rows = 12
    index = np.arange(rows, dtype=np.float32)
    base = pd.DataFrame(
        {
            "id": [f"row_{row}" for row in range(rows)],
            "well": ["well_a"] * rows,
            "well_row_idx": np.arange(rows),
            "outer_fold": [0] * rows,
            "md_since": index * 10.0,
        }
    )
    paths = np.column_stack(
        [
            100.0 + index,
            100.0 + index * 1.1,
            101.0 + index + np.sin(index),
            99.0 + index * 0.9,
            100.0 + index + (index >= 6) * 2.0,
            102.0 + index * 1.05,
        ]
    ).astype(np.float32)
    if shared_translation is not None:
        paths += np.asarray(shared_translation, dtype=np.float32)[:, None]
    confidence: dict[str, pd.DataFrame] = {}
    for candidate_index, candidate_id in enumerate(PRIMITIVES):
        frame = base.copy()
        frame["confidence_valid"] = True
        frame["sigma_tvt"] = 1.5 + candidate_index * 0.1
        frame["entropy"] = 0.1 + candidate_index * 0.01
        frame["support_count"] = 10 + candidate_index
        confidence[candidate_id] = frame
    return PrimitiveFold(base, paths, np.ones_like(paths, dtype=bool), confidence, PRIMITIVES)


def _write_raw(raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "MD": np.arange(12) * 10.0,
            "X": np.linspace(0.0, 2.0, 12),
            "GR": np.linspace(50.0, 70.0, 12),
            "TVT": np.linspace(100.0, 200.0, 12),
            "TVT_input": [1.0] * 3 + [np.nan] * 9,
        }
    ).to_csv(raw_dir / "well_a__horizontal_well.csv", index=False)


def test_contract_has_all_15_primitive_pairs() -> None:
    contract = _contract()
    validate_regime_contract(contract)
    assert len(primitive_ids(contract)) == 6
    assert len(pair_ids(primitive_ids(contract))) == 15


def test_block_features_are_target_free_and_translation_invariant(tmp_path: Path) -> None:
    raw_dir = tmp_path / "train"
    _write_raw(raw_dir)
    config = _feature_config()
    blocks, row_map = build_block_fingerprints(_bundle(), raw_dir, config)
    translated, _ = build_block_fingerprints(_bundle(np.full(12, 10.0)), raw_dir, config)
    columns = feature_columns(blocks)
    pair_and_bank = [
        column
        for column in columns
        if column.startswith(("pair__", "bank__")) and not column.endswith("first_diff_correlation")
    ]
    assert len([column for column in columns if column.startswith("pair__")]) == 15 * 12
    sigma_columns = [column for column in columns if "__sigma_tvt__" in column]
    assert len(sigma_columns) == 6
    np.testing.assert_allclose(
        blocks[pair_and_bank].to_numpy(),
        translated[pair_and_bank].to_numpy(),
        rtol=1e-5,
        atol=1e-3,
        equal_nan=True,
    )
    assert len(blocks) == 3
    assert len(row_map) == 12


def test_target_free_schema_allows_sigma_tvt_but_rejects_absolute_tvt() -> None:
    forbidden = [
        "tvt",
        "tvt_input",
        "true_tvt",
        "candidate_tvt",
        "last_known_tvt",
        "target",
        "actual_abs_error",
    ]
    assert_target_free_feature_schema(
        ["confidence__beam_mean__sigma_tvt__median", "pair__a__vs__b__gap_mean"],
        forbidden,
    )
    with pytest.raises(ValueError, match="forbidden regime features"):
        assert_target_free_feature_schema(["raw__tvt__mean", "pair__a__vs__b__gap_mean"], forbidden)
    with pytest.raises(ValueError, match="forbidden regime features"):
        assert_target_free_feature_schema(
            ["candidate__candidate_tvt", "pair__a__vs__b__gap_mean"], forbidden
        )


def test_outer_fold_regime_fit_is_deterministic() -> None:
    rng = np.random.default_rng(7)
    records = []
    for fold in range(5):
        for regime in range(3):
            for block in range(12):
                records.append(
                    {
                        "block_key": f"{fold}_{regime}_{block}",
                        "well": f"well_{fold}_{regime}_{block}",
                        "block_id": 0,
                        "outer_fold": fold,
                        "rows": 32,
                        "well_row_idx_start": 0,
                        "well_row_idx_end": 31,
                        "pair__a__vs__b__gap_mean": regime * 8.0 + rng.normal(0, 0.2),
                        "bank__range_mean": regime * 3.0 + rng.normal(0, 0.1),
                        "raw__gr__std": regime + rng.normal(0, 0.05),
                    }
                )
    blocks = pd.DataFrame(records)
    config = {
        "features": {"forbidden_feature_tokens": ["tvt", "target", "error"]},
        "regime": {
            "n_clusters": 3,
            "kmeans_n_init": 10,
            "quantile_range": [25.0, 75.0],
        },
        "reproducibility": {"seed": 42},
    }
    first, centroids, stability = fit_outer_fold_regimes(blocks, config)
    second, _, _ = fit_outer_fold_regimes(blocks, config)
    np.testing.assert_array_equal(first["regime"], second["regime"])
    np.testing.assert_allclose(
        first[[f"regime_probability_{index}" for index in range(3)]].sum(axis=1), 1.0
    )
    assert len(centroids) == 15
    assert len(stability) == 5


def test_post_assignment_score_audit_detects_regime_specific_winners() -> None:
    block_records = []
    row_records = []
    score_records = []
    for fold in range(5):
        for regime in range(3):
            block_key = f"block_{fold}_{regime}"
            well = f"well_{fold}_{regime}"
            block_records.append(
                {
                    "block_key": block_key,
                    "well": well,
                    "block_id": 0,
                    "outer_fold": fold,
                    "rows": 5,
                    "well_row_idx_start": 0,
                    "well_row_idx_end": 4,
                    "regime": regime,
                }
            )
            for row in range(5):
                row_id = f"row_{fold}_{regime}_{row}"
                row_records.append(
                    {
                        "id": row_id,
                        "well": well,
                        "well_row_idx": row,
                        "outer_fold": fold,
                        "block_key": block_key,
                        "regime": regime,
                    }
                )
                for candidate_index, candidate_id in enumerate(PRIMITIVES):
                    actual = 1.0 if candidate_index == regime else 4.0 + candidate_index
                    score_records.append(
                        {
                            "id": row_id,
                            "well": well,
                            "well_row_idx": row,
                            "outer_fold": fold,
                            "candidate_id": candidate_id,
                            "candidate_available": True,
                            "pred_abs_error": actual,
                            "actual_abs_error": actual,
                        }
                    )
    stability = pd.DataFrame(
        {
            "outer_fold": range(5),
            "outer_valid_blocks": [3] * 5,
            "centroid_matched_assignment_agreement": [1.0] * 5,
        }
    )
    config = {
        "regime": {"n_clusters": 3},
        "guards": {
            "occupancy": {
                "min_wells_per_regime_fold": 1,
                "min_block_share_per_regime_fold": 0.10,
                "min_passing_folds_per_regime": 4,
            },
            "stability": {"min_centroid_matched_assignment_agreement": 0.70},
            "separability": {
                "min_distinct_best_candidate_families": 2,
                "min_global_calibration_bias_range_ft": 0.25,
            },
        },
    }
    occupancy, metrics, result = evaluate_regime_separability(
        pd.DataFrame(block_records),
        pd.DataFrame(row_records),
        pd.DataFrame(score_records),
        _contract(),
        config,
        stability,
    )
    assert occupancy["guard_pass"].all()
    assert len(metrics.query("outer_fold == 'all'")) == 18
    assert result["separability"]["distinct_best_candidate_families"] == 3
    assert result["stage0_guard_pass"] is True


def test_streaming_score_audit_matches_in_memory_guard(tmp_path: Path) -> None:
    block_records = []
    row_records = []
    score_records = []
    for fold in range(5):
        for regime in range(3):
            block_key = f"block_{fold}_{regime}"
            well = f"well_{fold}_{regime}"
            block_records.append(
                {
                    "block_key": block_key,
                    "well": well,
                    "block_id": 0,
                    "outer_fold": fold,
                    "rows": 1,
                    "well_row_idx_start": 0,
                    "well_row_idx_end": 0,
                    "regime": regime,
                }
            )
            row_id = f"row_{fold}_{regime}"
            row_records.append(
                {
                    "id": row_id,
                    "well": well,
                    "well_row_idx": 0,
                    "outer_fold": fold,
                    "block_key": block_key,
                    "regime": regime,
                }
            )
            for candidate_index, candidate_id in enumerate(PRIMITIVES):
                actual = 1.0 if candidate_index == regime else 5.0
                score_records.append(
                    {
                        "id": row_id,
                        "well": well,
                        "well_row_idx": 0,
                        "outer_fold": fold,
                        "candidate_id": candidate_id,
                        "candidate_available": True,
                        "pred_abs_error": actual,
                        "actual_abs_error": actual,
                    }
                )
    score_path = tmp_path / "candidate_score_oof.parquet"
    pd.DataFrame(score_records).to_parquet(score_path, index=False)
    stability = pd.DataFrame(
        {
            "outer_fold": range(5),
            "outer_valid_blocks": [3] * 5,
            "centroid_matched_assignment_agreement": [1.0] * 5,
        }
    )
    config = {
        "regime": {"n_clusters": 3},
        "guards": {
            "occupancy": {
                "min_wells_per_regime_fold": 1,
                "min_block_share_per_regime_fold": 0.10,
                "min_passing_folds_per_regime": 4,
            },
            "stability": {"min_centroid_matched_assignment_agreement": 0.70},
            "separability": {
                "min_distinct_best_candidate_families": 2,
                "min_global_calibration_bias_range_ft": 0.25,
            },
        },
    }
    occupancy, metrics, result = evaluate_regime_separability_from_parquet(
        pd.DataFrame(block_records),
        pd.DataFrame(row_records),
        score_path,
        _contract(),
        config,
        stability,
        batch_size=7,
    )
    assert occupancy["guard_pass"].all()
    assert len(metrics.query("outer_fold == 'all'")) == 18
    assert result["stage0_guard_pass"] is True
    assert result["parquet_batches"] > 1
