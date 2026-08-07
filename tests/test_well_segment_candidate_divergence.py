from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.candidate_pairwise_regime import PrimitiveFold
from src.well_segment_candidate_divergence import (
    SEGMENTS,
    SEMANTIC_CLUSTERS,
    SIGNATURE_METRICS,
    assert_target_free_signature_schema,
    build_well_segment_signatures,
    evaluate_post_assignment_scores_from_parquet,
    fit_outer_fold_clusters,
    signature_feature_columns,
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


def _feature_config() -> dict:
    return {
        "features": {
            "segment_boundaries": [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0],
            "forbidden_feature_tokens": [
                "tvt",
                "candidate_tvt",
                "target",
                "error",
                "oracle",
                "md",
                "block",
            ],
        }
    }


def _bundle(common_translation: np.ndarray | None = None) -> PrimitiveFold:
    rows = 18
    wells = np.repeat(["well_a", "well_b"], rows // 2)
    row_index = np.tile(np.arange(rows // 2), 2)
    progress = row_index.astype(np.float32) / np.float32(rows // 2)
    base = pd.DataFrame(
        {
            "id": [f"{well}_{row}" for well, row in zip(wells, row_index, strict=True)],
            "well": wells,
            "well_row_idx": row_index,
            "outer_fold": np.zeros(rows, dtype=np.int8),
            "md_since": row_index.astype(np.float32),
        }
    )
    candidate_scale = np.arange(6, dtype=np.float32)[None, :]
    values = 1000.0 + row_index[:, None] + candidate_scale * progress[:, None] * 12.0
    if common_translation is not None:
        values = values + np.asarray(common_translation, dtype=np.float32)[:, None]
    return PrimitiveFold(
        base=base,
        values=np.asarray(values, dtype=np.float32),
        available=np.ones((rows, 6), dtype=bool),
        confidence={},
        primitive_ids=PRIMITIVES,
    )


def test_well_signature_is_18_features_target_free_and_translation_invariant() -> None:
    original, coverage = build_well_segment_signatures(_bundle(), _feature_config())
    translation = np.linspace(-500.0, 800.0, 18, dtype=np.float32)
    translated, _ = build_well_segment_signatures(
        _bundle(common_translation=translation), _feature_config()
    )
    columns = signature_feature_columns()
    assert len(columns) == 18
    assert_target_free_signature_schema(
        columns, _feature_config()["features"]["forbidden_feature_tokens"]
    )
    np.testing.assert_allclose(
        original[columns].to_numpy(),
        translated[columns].to_numpy(),
        rtol=1e-5,
        atol=2e-3,
        equal_nan=True,
    )
    assert len(coverage) == 2 * 3
    assert coverage["segment_rows"].eq(3).all()
    assert not coverage["fallback_required"].any()
    assert (
        original["segment__early__bank_range_mean"]
        < original["segment__middle__bank_range_mean"]
    ).all()
    assert (
        original["segment__middle__bank_range_mean"]
        < original["segment__late__bank_range_mean"]
    ).all()


def _synthetic_signatures() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    rng = np.random.default_rng(17)
    records = []
    coverage_records = []
    wells_per_cluster_fold = 12
    for fold in range(5):
        for regime in range(3):
            for index in range(wells_per_cluster_fold):
                well = f"well_{fold}_{regime}_{index}"
                record = {"well": well, "outer_fold": fold, "eval_rows": 90}
                for segment_index, segment in enumerate(SEGMENTS):
                    for metric_index, metric in enumerate(SIGNATURE_METRICS):
                        record[f"segment__{segment}__{metric}"] = (
                            regime * 12.0
                            + segment_index * 1.5
                            + metric_index * 0.05
                            + rng.normal(0.0, 0.03)
                        )
                    coverage_records.append(
                        {
                            "well": well,
                            "outer_fold": fold,
                            "segment": segment,
                            "segment_rows": 30,
                            "fallback_required": False,
                        }
                    )
                records.append(record)
    signatures = pd.DataFrame.from_records(records)
    coverage = pd.DataFrame.from_records(coverage_records)
    config = {
        "features": {"forbidden_feature_tokens": ["tvt", "target", "error", "md", "block"]},
        "cluster": {
            "n_clusters": 3,
            "quantile_range": [25.0, 75.0],
            "scaled_clip": [-10.0, 10.0],
            "kmeans_n_init": 10,
        },
        "reproducibility": {"seed": 42},
        "guards": {
            "technical": {
                "expected_wells": len(signatures),
                "expected_folds": 5,
                "expected_features": 18,
            },
            "occupancy": {
                "min_wells_per_cluster_outer_valid": 10,
                "min_wells_per_cluster_pooled": 50,
            },
            "stability": {
                "min_centroid_matched_assignment_agreement_each_fold": 0.95
            },
        },
    }
    return signatures, coverage, config


def test_outer_fold_cluster_is_deterministic_soft_and_semantic() -> None:
    signatures, coverage, config = _synthetic_signatures()
    first = fit_outer_fold_clusters(signatures, coverage, config)
    second = fit_outer_fold_clusters(signatures, coverage, config)
    assignments, centroids, preprocessors, stability, occupancy, profiles, summary = first
    np.testing.assert_array_equal(assignments["cluster_index"], second[0]["cluster_index"])
    probability_columns = [f"cluster_probability_{name}" for name in SEMANTIC_CLUSTERS]
    np.testing.assert_allclose(assignments[probability_columns].sum(axis=1), 1.0, atol=1e-6)
    assert set(assignments["semantic_cluster"]) == set(SEMANTIC_CLUSTERS)
    assert len(centroids) == 15
    assert len(preprocessors) == 5
    assert stability["centroid_matched_assignment_agreement"].ge(0.95).all()
    assert occupancy["guard_pass"].all()
    assert len(profiles) == 5 * 3 * 3
    assert summary["structure_guard_pass"] is True


def test_post_assignment_score_guard_uses_frozen_well_clusters(tmp_path: Path) -> None:
    assignments = []
    score_records = []
    best_by_cluster = {"low": 0, "middle": 1, "high": 2}
    difficulty = {"low": 1.0, "middle": 4.0, "high": 8.0}
    for fold in range(5):
        for cluster in SEMANTIC_CLUSTERS:
            for well_index in range(2):
                well = f"well_{fold}_{cluster}_{well_index}"
                assignments.append(
                    {
                        "well": well,
                        "outer_fold": fold,
                        "eval_rows": 1,
                        "semantic_cluster": cluster,
                    }
                )
                for candidate_index, candidate_id in enumerate(PRIMITIVES):
                    actual = (
                        difficulty[cluster]
                        + abs(candidate_index - best_by_cluster[cluster])
                        + well_index * 0.1
                    )
                    calibration = {"low": -0.5, "middle": 0.0, "high": 0.5}[cluster]
                    score_records.append(
                        {
                            "well": well,
                            "outer_fold": fold,
                            "candidate_id": candidate_id,
                            "candidate_available": True,
                            "actual_abs_error": actual,
                            "pred_abs_error": actual + calibration,
                        }
                    )
    score_path = tmp_path / "candidate_score_oof.parquet"
    pd.DataFrame(score_records).to_parquet(score_path, index=False)
    config = {
        "guards": {
            "technical": {"expected_rows": 30, "expected_wells": 30},
            "score": {
                "min_consistent_folds": 4,
                "min_distinct_candidates_in_modal_winner_pattern": 2,
                "calibration_epsilon": 1e-9,
            }
        }
    }
    metrics, calibration, well_metrics, summary = (
        evaluate_post_assignment_scores_from_parquet(
            pd.DataFrame(assignments),
            score_path,
            _contract(),
            config,
            {"structure_guard_pass": True},
            batch_size=13,
        )
    )
    assert len(metrics.query("outer_fold == 'all'")) == 18
    assert len(calibration) == len(metrics)
    assert len(well_metrics) == 30
    assert summary["candidate_winner"]["guard_pass"] is True
    assert summary["calibration_direction"]["guard_pass"] is True
    assert summary["worst_cluster_single_well"]["guard_pass"] is True
    assert summary["stage_a_guard_pass"] is True


def test_exp267_contract_disables_post_audit_training_and_inference() -> None:
    root = Path(__file__).resolve().parents[1]
    exp = root / "experiments" / (
        "exp267_well_segment_candidate_divergence_signature_cluster_on_exp265"
    )
    config = yaml.safe_load((exp / "config.yaml").read_text())
    assert config["experiment"]["route"] == "ensemble"
    assert config["execution"]["run_approved"] is False
    assert config["execution"]["stage_a_total_boosters"] == 0
    assert config["model"]["conditional_stage_b"]["enabled"] is False
    assert config["model"]["conditional_stage_b"]["planned_cpu_boosters"] == 10
    assert config["execution"]["inference_enabled"] is False
    assert config["features"]["expected_feature_count"] == 18
