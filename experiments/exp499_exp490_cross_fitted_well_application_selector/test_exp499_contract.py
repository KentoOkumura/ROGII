from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


MODULE_PATH = Path(__file__).with_name(
    "exp499_exp490_cross_fitted_well_application_selector_compact_selfcontained_train.py"
)
SPEC = importlib.util.spec_from_file_location("exp499_train", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def synthetic_predictions() -> pd.DataFrame:
    rows = []
    for well_index, well in enumerate(("00000001", "00000002")):
        for offset in range(4):
            exp226 = 100.0 + well_index + offset
            parent = exp226 + (well_index + 1) * 0.5
            candidate = exp226 + 0.2 * offset
            rows.append(
                {
                    "well": well,
                    "row_idx": offset + 10,
                    "suffix_offset": offset,
                    "tvt_geop": exp226 - 0.3,
                    "geometry_mean_reverting_hmm": candidate,
                    "geometry_mean_reverting_delta_mean": candidate - (exp226 - 0.3),
                    "geometry_mean_reverting_hmm_std": 0.5 + 0.1 * offset,
                    "exp357_parent_prediction": parent,
                    "exp226_pred": exp226,
                }
            )
    return pd.DataFrame(rows)


def synthetic_exp498() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "well": ["00000001", "00000002"],
            "rows": [4, 4],
            "visible_prefix_rows": [10, 12],
            "suffix_horizon_md": [4.0, 4.0],
            "k16_median_segment_span_ft": [1.0, 1.0],
            "prefix_gr_sigma": [20.0, 30.0],
            "prefix_gr_information_ratio": [2.0, 1.5],
            "geometry_disagreement_median_ft": [0.3, 0.4],
            "early_abs_offset_ft": [0.1, 0.2],
            "state_uncertainty_median_ft": [0.5, 0.6],
        }
    )


def test_feature_contract_has_exact_32_target_free_features() -> None:
    features = MODULE.build_target_free_feature_table(synthetic_predictions(), synthetic_exp498())
    assert features.columns.tolist() == ["well", *MODULE.FEATURE_COLUMNS]
    assert len(MODULE.FEATURE_COLUMNS) == 32
    assert features.shape == (2, 33)
    assert np.isfinite(features.loc[:, MODULE.FEATURE_COLUMNS].to_numpy()).all()
    assert (features.loc[:, MODULE.FEATURE_COLUMNS].to_numpy() >= 0).all()


def test_phase_a_rejects_outcome_column() -> None:
    predictions = synthetic_predictions()
    predictions["fold"] = 0
    try:
        MODULE.build_target_free_feature_table(predictions, synthetic_exp498())
    except ValueError as exc:
        assert "forbidden" in str(exc)
    else:
        raise AssertionError("Phase A accepted fold")


def test_policy_rmse_uses_row_weighted_candidate_or_parent() -> None:
    frame = pd.DataFrame(
        {
            "rows": [1, 3],
            "candidate_rmse_ft": [1.0, 5.0],
            "exp357_parent_rmse_ft": [3.0, 2.0],
        }
    )
    actual = MODULE.policy_rmse(frame, np.array([True, False]))
    expected = np.sqrt((1.0 * 1.0**2 + 3.0 * 2.0**2) / 4.0)
    assert abs(actual - expected) < 1e-12


def test_config_keeps_forbidden_execution_zero() -> None:
    config = MODULE.load_config(Path(__file__).with_name("config.yaml"))
    MODULE.validate_immutable_config(config)
    execution = config["execution_contract"]
    assert execution["new_hmm_well_runs"] == 0
    assert execution["parent_control_retraining"] == 0
    assert execution["gpu_runs"] == 0
    assert execution["maximum_total_cpu_model_fits"] == 45

