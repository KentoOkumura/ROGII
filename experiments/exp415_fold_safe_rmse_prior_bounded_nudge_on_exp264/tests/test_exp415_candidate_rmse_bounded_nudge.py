from __future__ import annotations

import importlib.util
import os
from collections import defaultdict
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest
import yaml

from src.candidate_rmse_bounded_nudge import (
    bounded_rmse_prior_policy,
    reconstruct_true_tvt,
    rmse_risk_certificate,
    validate_candidate_rmse_bounded_nudge_config,
)
from tests.test_support import require_saved_files

ROOT = Path(__file__).resolve().parents[3]
EXP = "exp415_fold_safe_rmse_prior_bounded_nudge_on_exp264"
EXP_DIR = ROOT / "experiments" / EXP
SOURCE_PATH = EXP_DIR / f"{EXP}_compact_selfcontained_train.py"


def _load_notebook_source() -> ModuleType:
    previous = os.environ.get("EXP415_IMPORT_ONLY")
    os.environ["EXP415_IMPORT_ONLY"] = "1"
    try:
        spec = importlib.util.spec_from_file_location("exp415_candidate", SOURCE_PATH)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if previous is None:
            os.environ.pop("EXP415_IMPORT_ONLY", None)
        else:
            os.environ["EXP415_IMPORT_ONLY"] = previous


def _policy_inputs() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[str],
    list[str],
]:
    candidates = ["a", "b", "fallback"]
    primary = ["a", "b"]
    score = np.asarray(
        [
            [0.0, 0.2, -99.0],
            [0.0, 0.0, -99.0],
            [0.0, 0.2, -99.0],
        ],
        dtype=np.float64,
    )
    tvt = np.asarray(
        [
            [100.0, 102.0, 1000.0],
            [200.0, 201.0, 1000.0],
            [300.0, 298.0, 1000.0],
        ],
        dtype=np.float64,
    )
    folds = np.asarray([0, 1, 0], dtype=np.int64)
    rmse = np.asarray(
        [
            [2.0, 1.0, 0.1],
            [1.0, 1.0, 0.1],
        ],
        dtype=np.float64,
    )
    return score, tvt, folds, rmse, candidates, primary


def test_exp415_static_and_zero_booster_contract() -> None:
    config = yaml.safe_load((EXP_DIR / "config.yaml").read_text())
    module = _load_notebook_source()
    static = module.validate_static_contract(config)
    assert config["experiment"]["route"] == "ensemble"
    assert static["run_approved"] is True
    assert config["implementation"]["canonical_notebook_adopted"] is True
    assert config["implementation"]["canonical_notebook_adoption_approved"] is True
    assert static["cost"] == {
        "active_variants": 1,
        "models": 0,
        "model_configs": 0,
        "folds_for_fit": 0,
        "boosters": 0,
        "control_retraining": 0,
        "pf_runs": 0,
        "hmm_runs": 0,
        "beam_runs": 0,
        "gpu_runs": 0,
        "inference_runs": 0,
        "submissions": 0,
    }
    assert validate_candidate_rmse_bounded_nudge_config(config["policy"]) == {
        "candidate_rmse_coefficient": 1.0,
        "blend_parent_weight": 0.5,
        "blend_prior_weight": 0.5,
        "max_abs_correction_ft": 0.25,
    }


def test_policy_uses_rmse_as_prior_and_clips_both_directions() -> None:
    score, tvt, folds, rmse, candidates, primary = _policy_inputs()
    result = bounded_rmse_prior_policy(
        score,
        tvt,
        folds,
        rmse,
        candidates,
        primary,
    )
    assert np.array_equal(result["parent_position"], [0, 0, 0])
    assert np.array_equal(result["prior_position"], [1, 0, 1])
    assert np.allclose(result["correction"], [0.25, 0.0, -0.25])
    assert np.allclose(result["prediction"], [100.25, 200.0, 299.75])
    assert float(np.max(np.abs(result["correction"]))) == 0.25


def test_notebook_policy_matches_reusable_implementation() -> None:
    module = _load_notebook_source()
    args = _policy_inputs()
    reusable = bounded_rmse_prior_policy(*args)
    notebook = module.bounded_rmse_prior_policy(*args)
    assert reusable.keys() == notebook.keys()
    for key in reusable:
        assert np.array_equal(reusable[key], notebook[key])


def test_truth_free_batch_rejects_actual_error_before_freeze() -> None:
    module = _load_notebook_source()
    config = yaml.safe_load((EXP_DIR / "config.yaml").read_text())
    candidates = config["candidate_bank"]["order"]
    rows = []
    for position, candidate_id in enumerate(candidates):
        rows.append(
            {
                "id": "row0",
                "well": "well0",
                "well_row_idx": 0,
                "outer_fold": 0,
                "md_since": 100.0,
                "candidate_id": candidate_id,
                "candidate_tvt": 1000.0 + position,
                "pred_abs_error": 0.1 + position,
                "actual_abs_error": 1.0 + position,
            }
        )
    frame = pd.DataFrame(rows)
    rmse = np.ones((5, len(candidates)), dtype=np.float64)
    with pytest.raises(ValueError, match="forbidden columns"):
        module.truth_free_policy_batch(frame, rmse, config)

    freeze = module.truth_free_policy_batch(
        frame.drop(columns="actual_abs_error"),
        rmse,
        config,
    )
    assert freeze["parent_candidate_tvt"].dtype == np.float64
    assert freeze["bounded_correction_ft"].dtype == np.float64
    assert freeze["bounded_prediction_tvt"].dtype == np.float64
    assert np.allclose(
        freeze["bounded_prediction_tvt"] - freeze["parent_candidate_tvt"],
        freeze["bounded_correction_ft"],
        atol=1.0e-12,
        rtol=0,
    )


def test_truth_reconstruction_is_exact_and_rejects_inconsistent_errors() -> None:
    truth = np.asarray([100.0, -50.0])
    candidates = np.asarray([[98.0, 101.0, 105.0], [-55.0, -49.0, -40.0]])
    errors = np.abs(candidates - truth[:, None])
    reconstructed, residual = reconstruct_true_tvt(candidates, errors)
    assert np.array_equal(reconstructed, truth)
    assert residual == 0.0
    errors[0, 2] += 0.1
    with pytest.raises(ValueError, match="reconstruction residual"):
        reconstruct_true_tvt(candidates, errors, tolerance=1.0e-6)


def test_minkowski_risk_certificate_bounds_any_scope() -> None:
    parent_error = np.asarray([2.0, -3.0, 5.0, -7.0])
    correction = np.asarray([0.25, -0.25, 0.1, 0.0])
    certificate = rmse_risk_certificate(
        parent_error,
        parent_error + correction,
        correction,
        correction_cap=0.25,
    )
    assert certificate["delta_lte_correction_rms"] is True
    assert certificate["correction_rms_lte_abs_max"] is True
    assert certificate["abs_max_lte_cap"] is True
    assert certificate["correction_abs_max"] == 0.25


def test_metric_aggregation_and_all_and_gate() -> None:
    module = _load_notebook_source()
    config = yaml.safe_load((EXP_DIR / "config.yaml").read_text())
    stats: dict[tuple[str, str], dict[str, float]] = defaultdict(
        lambda: {
            "rows": 0,
            "parent_sse": 0.0,
            "new_sse": 0.0,
            "correction_sq_sum": 0.0,
            "correction_abs_max": 0.0,
        }
    )
    keys = [
        ("overall", "overall"),
        *(("outer_fold", str(fold)) for fold in range(5)),
        *(("distance_bucket", name) for name, _, _ in module.DISTANCE_BUCKETS),
        ("hidden_like", "hidden_like_spatial"),
        ("hidden_like", "hidden_like_typewell_purged"),
        ("well", "well_good"),
        ("well", "well_risk"),
    ]
    for key in keys:
        stats[key] = {
            "rows": 100,
            "parent_sse": 10000.0,
            "new_sse": 9801.0,
            "correction_sq_sum": 6.25,
            "correction_abs_max": 0.25,
        }
    metrics = module.metric_frame(
        stats,
        tolerance=1.0e-12,
        correction_cap=0.25,
    )
    gate = module.evaluate_gate(
        metrics,
        {"synthetic_technical_check": True},
        config,
    )
    assert gate["technical"]["passed"] is True
    assert gate["scientific"]["passed"] is True
    assert gate["decision"] == "rmse_prior_bounded_nudge_method_confirmed_on_saved_oof"
    failed = module.evaluate_gate(
        metrics.assign(
            delta_lte_correction_rms=False,
        ),
        {"synthetic_technical_check": True},
        config,
    )
    assert failed["scientific"]["passed"] is False


def test_real_candidate_rmse_table_is_fold_safe_and_uses_no_weight_column() -> None:
    module = _load_notebook_source()
    config = yaml.safe_load((EXP_DIR / "config.yaml").read_text())
    path = (
        ROOT
        / "experiments"
        / "exp407_fold_safe_inverse_rmse_weighted_dual_selector_on_exp264"
        / "kaggle"
        / "output"
        / "train_v1_small"
        / "artifacts"
        / "candidate_task_weight_by_fold.csv"
    )
    require_saved_files(path)
    matrix, table, audit = module.load_candidate_rmse_matrix(path, config)
    assert matrix.shape == (5, 12)
    assert len(table) == 60
    assert audit["policy_columns_used"] == ["fit_candidate_rmse"]
    assert audit["weight_columns_used"] == []
