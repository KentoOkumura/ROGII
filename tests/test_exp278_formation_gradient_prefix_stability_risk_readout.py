from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "experiments"
    / "exp278_formation_gradient_prefix_stability_risk_readout_on_exp273"
    / "exp278_formation_gradient_prefix_stability_risk_readout_on_exp273_train.py"
)
SPEC = importlib.util.spec_from_file_location("exp278_train", SOURCE)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def synthetic_plane_frame(*, nearly_straight: bool = False) -> pd.DataFrame:
    if nearly_straight:
        x_grid, y_grid = np.meshgrid(np.linspace(-20, 20, 12), np.linspace(-15, 15, 10))
        x = x_grid.ravel()
        y = 1.0e-4 * x + 1.0e-8 * np.square(x)
    else:
        theta = np.linspace(0.0, 4.0 * np.pi, 120, endpoint=False)
        radius = np.linspace(10.0, 25.0, 120)
        x = radius * np.cos(theta)
        y = 0.8 * radius * np.sin(theta)
    z = 100.0 + 0.1 * x
    surface = 2.0 * x - 3.0 * y + 500.0
    return pd.DataFrame(
        {
            "MD": np.arange(len(x), dtype=float),
            "X": x,
            "Y": y,
            "Z": z,
            "TVT_input": surface - z,
        }
    )


def diagnostic_rows() -> pd.DataFrame:
    rows = []
    values = {
        "full": (1, 1, 1.0, 0.0, 1.0, 0.4, 2.5),
        "last512": (0, 1, 0.8, 0.2, 0.7, 0.2, 5.0),
        "last256": (0, 1, 0.5, 0.4, 0.5, 0.1, 10.0),
    }
    for well in ("well_a", "well_b"):
        for window, payload in values.items():
            valid, fit_valid, gx, gy, rmse, rank, condition = payload
            rows.append(
                {
                    "well": well,
                    "window": window,
                    "gradient_valid": valid,
                    "diagnostic_fit_valid": fit_valid,
                    "gradient_x": gx,
                    "gradient_y": gy,
                    "gradient_magnitude": float(np.hypot(gx, gy)),
                    "plane_rmse": rmse,
                    "xy_rank_ratio": rank,
                    "xy_condition_number": condition,
                }
            )
    return pd.DataFrame(rows)


def test_stable_outer_fold_is_hash_reproducible() -> None:
    assert MODULE.EXECUTE_NOTEBOOK is False
    first = MODULE.stable_outer_fold("00bbac68", 5)
    second = MODULE.stable_outer_fold("00bbac68", 5)
    assert first == second == 2
    assert 0 <= MODULE.stable_outer_fold("dd7d638e", 5) < 5


def test_huber_plane_recovers_two_dimensional_gradient() -> None:
    result = MODULE.fit_formation_plane(synthetic_plane_frame(), MODULE.CONFIG["plane"])
    assert result["valid"] is True
    assert result["diagnostic_fit_valid"] is True
    assert result["gradient_x"] == pytest.approx(2.0, abs=1.0e-10)
    assert result["gradient_y"] == pytest.approx(-3.0, abs=1.0e-10)
    assert result["generation_gradient_x"] == pytest.approx(2.0, abs=1.0e-10)
    assert result["plane_rmse"] < 1.0e-10


def test_diagnostic_plane_survives_generation_geometry_guard() -> None:
    result = MODULE.fit_formation_plane(
        synthetic_plane_frame(nearly_straight=True), MODULE.CONFIG["plane"]
    )
    assert result["valid"] is False
    assert result["diagnostic_fit_valid"] is True
    assert "rank_ratio_below_guard" in result["fallback_reason"]
    assert abs(result["gradient_x"]) > 0.1
    assert result["generation_gradient_x"] == 0.0
    assert result["generation_gradient_y"] == 0.0
    assert np.isfinite(result["plane_rmse"])
    assert np.isnan(result["generation_plane_rmse"])


def test_stability_risk_is_fixed_equal_weight_and_target_free() -> None:
    features = MODULE.build_prefix_stability_features(
        diagnostic_rows(), MODULE.CONFIG["risk"], n_folds=5
    )
    components = MODULE.CONFIG["risk"]["components"]
    expected = features[components].mean(axis=1)
    np.testing.assert_allclose(features["stability_risk_score"], expected)
    assert features["validity_flip"].eq(1.0).all()
    assert not any(
        token in column.lower()
        for column in features.columns
        for token in ("target", "true_tvt", "delta_rmse", "oracle", "error")
    )


def test_outcomes_attach_only_after_feature_freeze() -> None:
    features = MODULE.build_prefix_stability_features(
        diagnostic_rows(), MODULE.CONFIG["risk"], n_folds=5
    )
    outcomes = pd.DataFrame(
        {
            "well": ["well_a", "well_b"],
            "gradient_bank_mean_delta_rmse_vs_scalar": [1.0, 2.0],
        }
    )
    candidate = pd.DataFrame(
        {
            "well": ["well_a", "well_b"],
            "candidate": ["hmm_grad_center", "hmm_grad_center"],
            "delta_rmse_vs_scalar": [1.0, 2.0],
        }
    )
    before = MODULE.logical_frame_sha256(features, ["well"])
    merged, _ = MODULE.attach_outcomes_after_feature_freeze(features, outcomes, candidate)
    assert MODULE.logical_frame_sha256(features, ["well"]) == before
    assert "gradient_bank_mean_delta_rmse_vs_scalar" in merged
    leaked = features.assign(delta_rmse_hint=0.0)
    with pytest.raises(ValueError, match="leaked"):
        MODULE.attach_outcomes_after_feature_freeze(leaked, outcomes, candidate)


def test_streamed_candidate_rmse_and_saved_parity(tmp_path: Path) -> None:
    candidates = ["candidate_a", "candidate_b"]
    shard = pd.DataFrame(
        {
            "well": ["w0", "w0", "w1", "w1"],
            "true_tvt": [0.0, 2.0, 10.0, 14.0],
            "candidate_a": [0.0, 4.0, 11.0, 13.0],
            "candidate_b": [1.0, 1.0, 12.0, 12.0],
        }
    )
    path = tmp_path / "shard.csv.gz"
    shard.to_csv(path, index=False, compression="gzip")
    recomputed, summary = MODULE.stream_shard_candidate_rmse(path, candidates, chunk_rows=2)
    assert summary["rows"] == 4
    assert summary["wells"] == 2
    saved = recomputed.rename(columns={"recomputed_rmse": "rmse"}).copy()
    saved["scalar_rmse"] = saved.groupby("well")["rmse"].transform("min") - 0.25
    saved["delta_rmse_vs_scalar"] = saved["rmse"] - saved["scalar_rmse"]
    outcomes, selected, parity = MODULE.build_candidate_outcomes(
        saved, recomputed, candidates, parity_atol=1.0e-12
    )
    assert len(selected) == 4
    assert parity["parity_pass"].all()
    expected = saved.groupby("well")["delta_rmse_vs_scalar"].mean().sort_index()
    actual = outcomes.set_index("well")["gradient_bank_mean_delta_rmse_vs_scalar"].sort_index()
    np.testing.assert_allclose(actual, expected)


def test_five_fold_primary_guard_uses_only_primary_outcome() -> None:
    rows = []
    candidate_rows = []
    for fold in range(5):
        for local in range(20):
            risk = (local + 1) / 21.0 + fold * 1.0e-4
            well = f"w{fold}_{local:02d}"
            rows.append(
                {
                    "well": well,
                    "outer_fold": fold,
                    "full_gradient_valid": 1,
                    "stability_risk_score": risk,
                    "gradient_bank_mean_delta_rmse_vs_scalar": 2.0 * risk,
                    "gradient_bank_max_delta_rmse_vs_scalar": -risk,
                }
            )
            candidate_rows.append(
                {
                    "well": well,
                    "outer_fold": fold,
                    "full_gradient_valid": 1,
                    "stability_risk_score": risk,
                    "candidate": "report_only_bad_direction",
                    "delta_rmse_vs_scalar": -risk,
                }
            )
    merged = pd.DataFrame(rows)
    candidate = pd.DataFrame(candidate_rows)
    correlations = MODULE.build_correlation_readout(merged, candidate)
    quintiles = MODULE.build_risk_quintile_readout(merged, 5)
    features = merged[["well", "outer_fold", "full_gradient_valid", "stability_risk_score"]]
    config = copy.deepcopy(MODULE.CONFIG)
    config["guards"]["technical"].update(
        expected_wells=100,
        expected_full_valid_wells=100,
        min_full_valid_wells_per_fold=15,
    )
    guard = MODULE.evaluate_primary_guard(features, correlations, quintiles, config)
    assert guard["technical_and_primary_guard_pass"] is True
    assert guard["positive_folds"] == 5
    report_only = correlations[correlations["candidate"].eq("report_only_bad_direction")]
    assert (report_only["spearman"] < 0).all()
