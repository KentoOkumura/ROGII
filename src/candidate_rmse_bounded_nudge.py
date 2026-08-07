from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def validate_candidate_rmse_bounded_nudge_config(
    config: Mapping[str, Any],
) -> dict[str, float]:
    """Validate the frozen exp415 policy and return its numeric parameters."""

    expected = {
        "candidate_rmse_coefficient": 1.0,
        "blend_parent_weight": 0.5,
        "blend_prior_weight": 0.5,
        "max_abs_correction_ft": 0.25,
    }
    values = {key: float(config[key]) for key in expected}
    if values != expected:
        raise ValueError(f"bounded-nudge policy changed: {values}")
    if [float(item) for item in config["correction_clip_ft"]] != [-0.25, 0.25]:
        raise ValueError("bounded-nudge correction clip changed")
    forbidden = {
        "inverse_rmse_weight",
        "candidate_rmse_as_feature",
        "actual_error_before_freeze",
        "candidate_rmse_coefficient_grid",
        "blend_weight_grid",
        "correction_cap_grid",
        "candidate_subset",
        "well_specific_threshold",
        "rescue_after_gate",
    }
    if {str(item) for item in config["forbidden"]} != forbidden:
        raise ValueError("bounded-nudge forbidden-operation contract changed")
    return values


def bounded_rmse_prior_policy(
    parent_score: np.ndarray,
    candidate_tvt: np.ndarray,
    outer_fold: np.ndarray,
    rmse_matrix: np.ndarray,
    candidate_order: Sequence[str],
    primary_domain: Sequence[str],
    *,
    rmse_coefficient: float = 1.0,
    blend_prior_weight: float = 0.5,
    max_abs_correction_ft: float = 0.25,
) -> dict[str, np.ndarray]:
    """Apply a fold-safe candidate-RMSE prior and a bounded TVT correction."""

    score = np.asarray(parent_score, dtype=np.float64)
    tvt = np.asarray(candidate_tvt, dtype=np.float64)
    folds = np.asarray(outer_fold, dtype=np.int64)
    rmse = np.asarray(rmse_matrix, dtype=np.float64)
    candidates = [str(item) for item in candidate_order]
    primary = [str(item) for item in primary_domain]

    if score.shape != tvt.shape or score.ndim != 2:
        raise ValueError("score and candidate TVT matrices must align")
    if score.shape[1] != len(candidates):
        raise ValueError("candidate matrix width changed")
    if folds.shape != (len(score),):
        raise ValueError("outer fold vector does not match the base rows")
    if rmse.ndim != 2 or rmse.shape[1] != len(candidates):
        raise ValueError("candidate RMSE matrix shape changed")
    if not primary or len(primary) != len(set(primary)):
        raise ValueError("primary candidate domain must be non-empty and unique")
    if any(name not in candidates for name in primary):
        raise ValueError("primary candidate domain contains an unknown candidate")
    if (
        not np.isfinite(score).all()
        or not np.isfinite(tvt).all()
        or not np.isfinite(rmse).all()
        or np.any(rmse <= 0)
    ):
        raise ValueError("bounded-nudge inputs contain invalid values")
    if np.any((folds < 0) | (folds >= rmse.shape[0])):
        raise ValueError("outer fold is outside the RMSE table")
    if not math.isfinite(rmse_coefficient) or rmse_coefficient < 0:
        raise ValueError("candidate RMSE coefficient must be finite and non-negative")
    if not math.isfinite(blend_prior_weight) or not 0 <= blend_prior_weight <= 1:
        raise ValueError("prior blend weight must be between zero and one")
    if not math.isfinite(max_abs_correction_ft) or max_abs_correction_ft <= 0:
        raise ValueError("correction cap must be finite and positive")

    positions = np.asarray([candidates.index(name) for name in primary], dtype=np.int64)
    parent_local = np.argmin(score[:, positions], axis=1)
    prior_score = score[:, positions] + float(rmse_coefficient) * rmse[folds][:, positions]
    prior_local = np.argmin(prior_score, axis=1)
    parent_position = positions[parent_local]
    prior_position = positions[prior_local]
    rows = np.arange(len(score))
    parent_tvt = tvt[rows, parent_position]
    prior_tvt = tvt[rows, prior_position]
    raw_nudge = float(blend_prior_weight) * (prior_tvt - parent_tvt)
    correction = np.clip(
        raw_nudge,
        -float(max_abs_correction_ft),
        float(max_abs_correction_ft),
    )
    prediction = parent_tvt + correction
    if not np.isfinite(prediction).all():
        raise ValueError("bounded RMSE-prior prediction is non-finite")
    if float(np.max(np.abs(correction))) > float(max_abs_correction_ft) + 1e-12:
        raise AssertionError("bounded correction exceeds its risk budget")
    return {
        "parent_position": parent_position,
        "prior_position": prior_position,
        "parent_tvt": parent_tvt,
        "prior_tvt": prior_tvt,
        "parent_score": score[rows, parent_position],
        "prior_parent_score": score[rows, prior_position],
        "prior_adjusted_score": prior_score[rows, prior_local],
        "raw_nudge": raw_nudge,
        "correction": correction,
        "prediction": prediction,
    }


def reconstruct_true_tvt(
    candidate_tvt: np.ndarray,
    actual_abs_error: np.ndarray,
    *,
    tolerance: float = 1.0e-6,
) -> tuple[np.ndarray, float]:
    """Recover row truth from candidate values and saved absolute errors."""

    values = np.asarray(candidate_tvt, dtype=np.float64)
    errors = np.asarray(actual_abs_error, dtype=np.float64)
    if values.shape != errors.shape or values.ndim != 2:
        raise ValueError("candidate values and actual errors must align")
    if not np.isfinite(values).all() or not np.isfinite(errors).all() or np.any(errors < 0):
        raise ValueError("truth reconstruction inputs are invalid")
    plus = values[:, 0] + errors[:, 0]
    minus = values[:, 0] - errors[:, 0]
    plus_residual = np.mean(np.abs(np.abs(values - plus[:, None]) - errors), axis=1)
    minus_residual = np.mean(np.abs(np.abs(values - minus[:, None]) - errors), axis=1)
    truth = np.where(plus_residual <= minus_residual, plus, minus)
    residual = np.abs(np.abs(values - truth[:, None]) - errors)
    max_residual = float(np.max(residual))
    if max_residual > float(tolerance):
        raise ValueError(f"true TVT reconstruction residual {max_residual} exceeds {tolerance}")
    return truth, max_residual


def rmse_risk_certificate(
    parent_error: np.ndarray,
    new_error: np.ndarray,
    correction: np.ndarray,
    *,
    correction_cap: float,
    tolerance: float = 1.0e-12,
) -> dict[str, float | bool | int]:
    """Check Minkowski's RMSE regression bound for one arbitrary scope."""

    parent = np.asarray(parent_error, dtype=np.float64)
    new = np.asarray(new_error, dtype=np.float64)
    delta = np.asarray(correction, dtype=np.float64)
    if parent.shape != new.shape or parent.shape != delta.shape or parent.ndim != 1:
        raise ValueError("risk-certificate vectors must be aligned and one-dimensional")
    if not len(parent) or not all(np.isfinite(item).all() for item in (parent, new, delta)):
        raise ValueError("risk-certificate vectors are empty or non-finite")
    if not np.allclose(new, parent + delta, atol=tolerance, rtol=0):
        raise ValueError("new error must equal parent error plus correction")
    parent_rmse = float(np.sqrt(np.mean(np.square(parent))))
    new_rmse = float(np.sqrt(np.mean(np.square(new))))
    correction_rms = float(np.sqrt(np.mean(np.square(delta))))
    correction_abs_max = float(np.max(np.abs(delta)))
    rmse_delta = new_rmse - parent_rmse
    return {
        "rows": len(parent),
        "parent_rmse": parent_rmse,
        "new_rmse": new_rmse,
        "delta_rmse_new_minus_parent": rmse_delta,
        "correction_rms": correction_rms,
        "correction_abs_max": correction_abs_max,
        "delta_lte_correction_rms": rmse_delta <= correction_rms + tolerance,
        "correction_rms_lte_abs_max": (correction_rms <= correction_abs_max + tolerance),
        "abs_max_lte_cap": correction_abs_max <= correction_cap + tolerance,
    }
