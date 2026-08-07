from __future__ import annotations

import importlib.util
import json
import os
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "experiments" / "exp292_typewell_gr_warp_rate_identifiability_audit"
MODULE_PATH = EXP_DIR / (
    "exp292_typewell_gr_warp_rate_identifiability_audit_compact_selfcontained_train.py"
)
os.environ["EXP292_IMPORT_ONLY"] = "1"
SPEC = importlib.util.spec_from_file_location("exp292_contract", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
EXP292 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = EXP292
SPEC.loader.exec_module(EXP292)


def load_config() -> dict:
    return yaml.safe_load((EXP_DIR / "config.yaml").read_text())


def synthetic_typewell() -> tuple[np.ndarray, np.ndarray]:
    tvt = np.linspace(0.0, 2000.0, 4001)
    gr = 70.0 + 20.0 * np.sin(tvt / 4.0) + 8.0 * np.cos(tvt / 11.0)
    return tvt, gr


def calibrated_prefix(config: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    typewell_tvt, typewell_gr = synthetic_typewell()
    tvt_input = np.linspace(100.0, 400.0, 500)
    reference = np.interp(tvt_input, typewell_tvt, typewell_gr)
    horizontal_gr = 1.7 * reference + 4.5
    return horizontal_gr, tvt_input, typewell_tvt, typewell_gr


def test_config_has_fixed_zero_booster_contract() -> None:
    config = load_config()
    EXP292.validate_scientific_contract(config)
    assert config["experiment"]["route"] == "pf_beam"
    assert config["execution"]["active_audit_variants"] == 1
    assert config["execution"]["lightgbm_config_count"] == 0
    assert config["execution"]["trained_fold_count"] == 0
    assert config["execution"]["total_boosters"] == 0
    assert config["execution"]["hmm_pf_well_runs"] == 0
    assert config["execution"]["control_or_parent_retraining"] is False
    assert config["execution"]["kaggle_push_approved"] is True
    assert config["execution"]["canonical_train_notebook_adopted"] is True
    assert config["inference"]["enabled"] is False

    unsafe = deepcopy(config)
    unsafe["audit"]["composite"]["weights"] = [0.5, 0.25, 0.25]
    with pytest.raises(ValueError, match="equal weights"):
        EXP292.validate_scientific_contract(unsafe)


def test_target_free_boundary_rejects_truth_and_error_columns() -> None:
    safe = pd.DataFrame({"well": ["a"], "hmm_ir_tail30": [1000.0], "GR": [70.0]})
    EXP292.validate_target_free_frame(safe)
    for column in ("TVT", "true_tvt", "target", "candidate_error", "oracle_rank"):
        with pytest.raises(ValueError, match="forbidden truth"):
            EXP292.validate_target_free_frame(safe.assign(**{column: 0.0}))


def test_typewell_preparation_is_stable_and_uses_duplicate_median() -> None:
    frame = pd.DataFrame({"TVT": [2.0, 1.0, 1.0, np.nan, 3.0], "GR": [20.0, 8.0, 12.0, 99.0, 30.0]})
    tvt, gr = EXP292.prepare_typewell_curve(frame)
    np.testing.assert_allclose(tvt, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(gr, [10.0, 20.0, 30.0])
    query = EXP292.interpolate_no_extrapolation(np.asarray([0.0, 1.5, 4.0]), tvt, gr)
    assert np.isnan(query[0]) and np.isnan(query[2])
    assert query[1] == pytest.approx(15.0)


def test_prefix_calibration_recovers_affine_map_and_clipped_scales() -> None:
    config = load_config()
    horizontal_gr, tvt_input, typewell_tvt, typewell_gr = calibrated_prefix(config)
    result = EXP292.robust_affine_calibration(
        horizontal_gr, tvt_input, typewell_tvt, typewell_gr, config
    )
    assert result.valid
    assert result.slope == pytest.approx(1.7, abs=1.0e-10)
    assert result.intercept == pytest.approx(4.5, abs=1.0e-9)
    assert result.sigma == 10.0
    assert result.derivative_sigma == 1.0
    assert result.pairs == 500

    invalid = EXP292.robust_affine_calibration(
        horizontal_gr[:20], tvt_input[:20], typewell_tvt, typewell_gr, config
    )
    assert not invalid.valid
    assert invalid.reason == "prefix_pairs_below_minimum"


def test_real_forward_gr_score_prefers_generating_candidate() -> None:
    config = load_config()
    horizontal_gr, tvt_input, typewell_tvt, typewell_gr = calibrated_prefix(config)
    calibration = EXP292.robust_affine_calibration(
        horizontal_gr, tvt_input, typewell_tvt, typewell_gr, config
    )
    rows = 256
    rates = [0.44, 0.50, 0.56, 0.62, 0.68]
    paths = {
        candidate: 500.0 + rate * np.arange(rows)
        for candidate, rate in zip(EXP292.CANDIDATES, rates, strict=True)
    }
    generating = "hmm_ir_w128"
    observed = (
        calibration.slope * np.interp(paths[generating], typewell_tvt, typewell_gr)
        + calibration.intercept
    )
    scores, meta = EXP292.score_candidate_horizon(
        observed, paths, typewell_tvt, typewell_gr, calibration, rows, config
    )
    selected = scores.sort_values(["composite", "candidate_index"], ascending=[False, True]).iloc[
        0
    ]["candidate"]
    assert selected == generating
    assert meta["effective_horizon_rows"] == 256
    assert meta["common_pairs"] == 256


def test_candidate_zscore_zero_mad_and_safe_first_tie() -> None:
    config = load_config()
    np.testing.assert_array_equal(EXP292.candidate_robust_zscore(np.ones(5), config), np.zeros(5))
    rows = pd.DataFrame(
        {
            "well": "well-a",
            "fold": 0,
            "horizon_rows": 256,
            "control": "real",
            "candidate": EXP292.CANDIDATES,
            "candidate_index": range(5),
            "composite": 1.0,
            "eligible": True,
            "eligibility_reason": "ok",
            "effective_horizon_rows": 256,
            "common_pairs": 256,
        }
    )
    selected = EXP292.select_target_free(rows)
    assert selected.iloc[0]["selected_candidate"] == EXP292.SAFE_CANDIDATE


def test_shuffle_is_stable_local_and_fails_when_range_is_empty() -> None:
    config = load_config()
    first = EXP292.stable_rotation_offset("well-a", 256, config)
    second = EXP292.stable_rotation_offset("well-a", 256, config)
    assert first == second
    assert 64 <= int(first) <= 192
    assert EXP292.stable_rotation_offset("well-a", 63, config) is None


def test_group_folds_are_deterministic_and_cover_every_well() -> None:
    wells = [f"well-{index:03d}" for index in range(23)]
    first = EXP292.assign_canonical_group_folds(reversed(wells), n_splits=5)
    second = EXP292.assign_canonical_group_folds(wells, n_splits=5)
    assert first == second
    assert set(first) == set(wells)
    assert set(first.values()) == set(range(5))


def test_exp268_aggregate_preflight_requires_diversity_and_frozen_shas(tmp_path: Path) -> None:
    config = load_config()
    summary_path = tmp_path / "summary.json"
    manifest_path = tmp_path / "manifest.csv"
    summary = {
        "status": "completed_train_side_candidate_bank_audit_pending_review",
        "rows": 3_783_989,
        "wells": 773,
        "direct_candidates": ["exp072_likpf_mean", *EXP292.CANDIDATES],
        "rate_spread": {"zero_rate_spread_wells": 0},
        "prediction_content_sha256": "synthetic",
    }
    summary_path.write_text(json.dumps(summary))
    specs = config["data"]["exp268_shards"]["shard_specs"]
    pd.DataFrame(
        {
            "role": ["shard0", "shard1"],
            "decompressed_sha256": [
                specs[0]["expected_decompressed_sha256"],
                specs[1]["expected_decompressed_sha256"],
            ],
        }
    ).to_csv(manifest_path, index=False)
    exp268 = config["data"]["exp268_shards"]
    exp268["aggregate_summary_filename"] = summary_path.name
    exp268["aggregate_manifest_filename"] = manifest_path.name
    exp268["aggregate_summary_candidates"] = [str(summary_path)]
    exp268["aggregate_manifest_candidates"] = [str(manifest_path)]
    exp268["expected_aggregate_summary_sha256"] = EXP292.sha256_path(summary_path)
    exp268["expected_aggregate_manifest_sha256"] = EXP292.sha256_path(manifest_path)
    exp268["expected_prediction_content_sha256"] = "synthetic"
    checked, records = EXP292.preflight_exp268_aggregate(config)
    assert checked["rows"] == 3_783_989
    assert len(records) == 2

    summary["rate_spread"]["zero_rate_spread_wells"] = 773
    summary_path.write_text(json.dumps(summary))
    exp268["expected_aggregate_summary_sha256"] = EXP292.sha256_path(summary_path)
    with pytest.raises(RuntimeError, match="diversity"):
        EXP292.preflight_exp268_aggregate(config)


def test_candidate_best_ties_are_all_positive() -> None:
    truth = np.asarray([1.0, 2.0, 3.0])
    paths = {candidate: truth.copy() for candidate in EXP292.CANDIDATES}
    labels = EXP292.best_candidate_labels(truth, paths, horizon=256, atol=1.0e-9)
    assert all(labels.values())


def test_inference_is_explicitly_fail_closed() -> None:
    source = (
        EXP_DIR
        / "exp292_typewell_gr_warp_rate_identifiability_audit_compact_selfcontained_inference.py"
    ).read_text()
    assert "raise RuntimeError" in source
    assert "decoder requires separate approval" in source
    assert "submission.csv" not in source
