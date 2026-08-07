from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.continuous_well_divergence_risk import (
    AXIS_COLUMNS,
    build_readout_tables,
    fit_oof_continuous_axes,
    primary_feature_columns,
    stream_candidate_well_metrics,
)
from src.well_segment_candidate_divergence import signature_feature_columns

CANDIDATES = [
    "exp226_k16",
    "selfgr_hmm_a070",
    "likpf_mean",
    "exact_hmm",
    "pf_ancc",
    "beam_mean",
]


def _axis_config(wells: int) -> dict:
    return {
        "axes": {"primary_feature_count": 12},
        "preprocessing": {
            "quantile_range": [25.0, 75.0],
            "scaled_clip": [-10.0, 10.0],
        },
        "guards": {
            "technical": {
                "expected_wells": wells,
                "expected_folds": 5,
                "expected_features": 18,
            }
        },
    }


def _synthetic_signatures(wells_per_fold: int = 8) -> pd.DataFrame:
    records = []
    features = signature_feature_columns()
    for fold in range(5):
        for index in range(wells_per_fold):
            divergence = 1.0 + index * 0.7 + fold * 0.05
            record = {
                "well": f"well_{fold}_{index:02d}",
                "outer_fold": fold,
                "eval_rows": 100 + index,
            }
            for feature_index, feature in enumerate(features):
                metric = feature.rsplit("__", 1)[-1]
                if metric in {
                    "bank_range_mean",
                    "bank_range_p90",
                    "pair_abs_gap_mean",
                    "pair_abs_gap_p90",
                }:
                    record[feature] = divergence + feature_index * 0.01
                else:
                    record[feature] = 0.5 * divergence + feature_index * 0.005
            records.append(record)
    return pd.DataFrame.from_records(records)


def test_oof_axes_are_deterministic_and_outer_valid_does_not_fit_preprocessor() -> None:
    signatures = _synthetic_signatures()
    config = _axis_config(len(signatures))
    first_axes, first_preprocessors = fit_oof_continuous_axes(signatures, config)
    second_axes, _ = fit_oof_continuous_axes(signatures, config)
    np.testing.assert_allclose(
        first_axes[list(AXIS_COLUMNS)].to_numpy(),
        second_axes[list(AXIS_COLUMNS)].to_numpy(),
        rtol=0.0,
        atol=0.0,
    )
    assert len(primary_feature_columns()) == 12
    assert first_axes["signature_imputed_values"].eq(0).all()

    modified = signatures.copy()
    modified.loc[modified["outer_fold"] == 0, signature_feature_columns()] += 1000.0
    _, modified_preprocessors = fit_oof_continuous_axes(modified, config)
    original_fold0 = first_preprocessors[0]
    modified_fold0 = modified_preprocessors[0]
    assert original_fold0["outer_fold"] == modified_fold0["outer_fold"] == 0
    np.testing.assert_allclose(
        original_fold0["robust_scaler_center"],
        modified_fold0["robust_scaler_center"],
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        original_fold0["pca_component_oriented"],
        modified_fold0["pca_component_oriented"],
        rtol=0.0,
        atol=0.0,
    )


def test_candidate_score_streaming_builds_complete_well_candidate_metrics(
    tmp_path: Path,
) -> None:
    signatures = _synthetic_signatures(wells_per_fold=2)
    rows = []
    for signature in signatures.itertuples(index=False):
        for candidate_index, candidate_id in enumerate(CANDIDATES):
            for row_index in range(3):
                actual = 1.0 + candidate_index * 0.2 + row_index * 0.1
                rows.append(
                    {
                        "well": signature.well,
                        "outer_fold": signature.outer_fold,
                        "candidate_id": candidate_id,
                        "candidate_available": True,
                        "actual_abs_error": actual,
                        "pred_abs_error": actual - 0.25,
                    }
                )
    score_path = tmp_path / "candidate_score_oof.parquet"
    pd.DataFrame.from_records(rows).to_parquet(score_path, index=False)
    metrics, evidence = stream_candidate_well_metrics(
        score_path,
        signatures,
        CANDIDATES,
        batch_size=17,
        expected_rows_per_candidate=len(signatures) * 3,
    )
    assert len(metrics) == len(signatures) * len(CANDIDATES)
    assert metrics.groupby("well")["candidate_id"].nunique().eq(6).all()
    assert np.allclose(metrics["calibration_bias"], -0.25)
    assert evidence["wells"] == len(signatures)
    assert set(evidence["rows_by_candidate"]) == set(CANDIDATES)


def test_primary_guard_uses_fold_direction_and_bootstrap_effect_floor() -> None:
    axes_records = []
    candidate_records = []
    for fold in range(5):
        for index in range(14):
            well = f"well_{fold}_{index:02d}"
            axis = -2.0 + 4.0 * index / 13.0 + fold * 0.01
            axes_records.append(
                {
                    "well": well,
                    "outer_fold": fold,
                    "eval_rows": 100,
                    "fixed_range_gap_axis": axis,
                    "pca1_axis": axis * 0.9,
                    "signature_imputed_values": 0,
                }
            )
            for candidate_index, candidate_id in enumerate(CANDIDATES):
                actual = 8.0 + 1.5 * axis + candidate_index * 0.03
                calibration = -0.8 * axis + candidate_index * 0.01
                candidate_records.append(
                    {
                        "well": well,
                        "outer_fold": fold,
                        "candidate_id": candidate_id,
                        "rows": 100,
                        "actual_mae": actual,
                        "predicted_abs_error_mean": actual + calibration,
                        "calibration_bias": calibration,
                    }
                )
    config = {
        "candidate_bank": {"primitive_ids": CANDIDATES},
        "bootstrap": {"seed": 272042, "n_resamples": 300, "interval": 0.95},
        "readout": {"axis_quantiles": 5},
        "guards": {
            "monotonic_risk": {
                "required_same_direction_folds": 5,
                "sign_epsilon": 1e-12,
                "min_actual_mae_bootstrap_lower": 0.05,
                "max_calibration_bias_bootstrap_upper": -0.05,
            }
        },
    }
    first = build_readout_tables(
        pd.DataFrame.from_records(axes_records),
        pd.DataFrame.from_records(candidate_records),
        config,
    )
    second = build_readout_tables(
        pd.DataFrame.from_records(axes_records),
        pd.DataFrame.from_records(candidate_records),
        config,
    )
    assert first["guard"]["continuous_risk_guard_pass"] is True
    assert first["guard"]["actual_mae_positive_fold_count"] == 5
    assert first["guard"]["calibration_bias_negative_fold_count"] == 5
    pd.testing.assert_frame_equal(first["bootstrap"], second["bootstrap"])


def test_exp272_config_keeps_zero_booster_and_inference_disabled() -> None:
    path = Path(
        "experiments/exp272_continuous_well_divergence_risk_readout_on_exp267/config.yaml"
    )
    config = yaml.safe_load(path.read_text())
    assert config["experiment"]["route"] == "ensemble"
    assert config["axes"]["primary"] == "fixed_range_gap_axis"
    assert config["axes"]["sensitivity_decision_role"] == (
        "report_only_cannot_rescue_primary_guard"
    )
    assert config["model"]["variants"] == 0
    assert config["model"]["lightgbm_configs"] == 0
    assert config["model"]["folds_trained"] == 0
    assert config["model"]["total_boosters"] == 0
    assert config["execution"]["run_approved"] is False
    assert config["execution"]["inference_enabled"] is False
    assert config["execution"]["submission_enabled"] is False
    assert config["inference"]["enabled"] is False
    assert config["inference"]["create_submission"] is False
