from __future__ import annotations

import importlib.util
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "experiments"
    / "exp350_exp345_bidirectional_gr_affine_smoother"
    / "exp350_exp345_bidirectional_gr_affine_smoother_compact_selfcontained_train.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("exp350_impl", SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_config(module):
    return module.read_yaml(SOURCE.parent / "config.yaml")


def test_scientific_contract_and_stage_0_lifecycle() -> None:
    module = load_module()
    config = load_config(module)
    module.validate_scientific_contract(config)
    assert config["experiment"]["status"] == "stage_0_failed_closed"
    assert config["execution"]["kaggle_push_approved"] is False
    assert config["execution"]["run_stage_0"] is False

    approved_run_config = deepcopy(config)
    approved_run_config["execution"]["kaggle_push_approved"] = True
    approved_run_config["execution"]["run_stage_0"] = True
    approved_run_config["execution"]["active_stage"] = "stage_0_full"
    module.validate_scientific_contract(
        approved_run_config,
        require_run_approval=True,
    )
    assert module.selected_stage(approved_run_config) == "stage_0_full"


def test_hidden_like_role_uses_valid_rows() -> None:
    module = load_module()
    actual = module.role_mask(
        pd.Series(["train", "valid", "purged_train_excluded", "VALID"])
    )
    np.testing.assert_array_equal(actual, [False, True, False, True])


def test_forward_schedule_parity_accepts_matching_nan_pattern() -> None:
    module = load_module()
    frame = pd.DataFrame(
        {
            "row_idx": [10, 11],
            "affine_scale_a": [1.0, 1.1],
            "affine_intercept_b": [0.0, 0.2],
            "raw_gr_update": [True, False],
            "predictive_nll_identity": [2.0, np.nan],
            "predictive_nll_affine": [1.5, np.nan],
        }
    )
    report = module.forward_schedule_parity(frame, frame.copy(), 1.0e-10)
    assert report["passed"] is True
    assert report["maximum_absolute_difference"] == 0.0


def test_rts_terminal_and_covariance_contract() -> None:
    module = load_module()
    config = load_config(module)
    forward = pd.DataFrame(
        {
            "row_idx": [0, 1, 2],
            "affine_scale_a": np.exp([0.0, 0.01, 0.02]),
            "affine_intercept_b": [0.0, 0.1, 0.2],
            "raw_gr_update": [True, True, True],
            "predictive_nll_identity": [1.0, 1.0, 1.0],
            "predictive_nll_affine": [0.9, 0.9, 0.9],
            "observation_variance": [100.0, 100.0, 100.0],
            "predicted_intercept_b": [0.0, 0.05, 0.15],
            "predicted_log_scale_a": [0.0, 0.005, 0.015],
            "predicted_p00": [1.1, 0.9, 0.8],
            "predicted_p01": [0.0, 0.0, 0.0],
            "predicted_p11": [0.11, 0.09, 0.08],
            "filtered_intercept_b": [0.0, 0.1, 0.2],
            "filtered_log_scale_a": [0.0, 0.01, 0.02],
            "filtered_p00": [0.8, 0.7, 0.6],
            "filtered_p01": [0.0, 0.0, 0.0],
            "filtered_p11": [0.08, 0.07, 0.06],
        }
    )
    context = {
        "initial_state": np.array([0.0, 0.0]),
        "initial_covariance": np.diag([1.0, 0.1]),
        "suffix_x": np.array([50.0, 55.0, 60.0]),
        "raw_gr": np.array([50.0, 55.0, 60.0]),
    }
    smoothed, audit = module.bidirectional_rts_schedule(
        forward, {"fallback": False}, context, config
    )
    assert smoothed.iloc[-1]["affine_intercept_b"] == 0.2
    assert smoothed.iloc[-1]["smoothed_log_scale_a"] == 0.02
    assert audit["terminal_state_max_abs_error"] == 0.0
    assert audit["terminal_covariance_max_abs_error"] == 0.0
    assert audit["covariance_minimum_eigenvalue_before_floor"] >= -1.0e-8
    assert audit["covariance_contraction_max_positive_eigenvalue"] <= 1.0e-8
