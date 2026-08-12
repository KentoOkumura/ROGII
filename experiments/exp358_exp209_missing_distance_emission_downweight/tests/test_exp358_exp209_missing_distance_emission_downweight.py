from __future__ import annotations

import importlib.util
import os
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = ROOT / "experiments" / "exp358_exp209_missing_distance_emission_downweight"
TRAIN_SOURCE = EXP_DIR / (
    "exp358_exp209_missing_distance_emission_downweight_compact_selfcontained_train.py"
)
STAGE1_SOURCE = EXP_DIR / (
    "exp358_exp209_missing_distance_emission_downweight_stage1_compact_selfcontained_train.py"
)
INFERENCE_SOURCE = EXP_DIR / (
    "exp358_exp209_missing_distance_emission_downweight_compact_selfcontained_inference.py"
)
PARENT_SOURCE = (
    ROOT
    / "experiments"
    / "exp209_exp072_exp205_joint_exact_parity_fast_cache_generation"
    / "exact_hmm_smoother.py"
)


def load_module(path: Path, name: str):
    previous = os.environ.get("EXP358_IMPORT_ONLY")
    os.environ["EXP358_IMPORT_ONLY"] = "1"
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        if previous is None:
            os.environ.pop("EXP358_IMPORT_ONLY", None)
        else:
            os.environ["EXP358_IMPORT_ONLY"] = previous


@pytest.fixture(scope="module")
def train():
    return load_module(TRAIN_SOURCE, "exp358_train_test")


@pytest.fixture(scope="module")
def stage1():
    return load_module(STAGE1_SOURCE, "exp358_stage1_train_test")


@pytest.fixture(scope="module")
def inference():
    return load_module(INFERENCE_SOURCE, "exp358_inference_test")


@pytest.fixture(scope="module")
def config():
    value = yaml.safe_load((EXP_DIR / "config.yaml").read_text())
    assert isinstance(value, dict)
    return value


def synthetic_well() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = 38
    known_rows = 30
    horizontal = pd.DataFrame(
        {
            "MD": np.arange(rows, dtype=float) * 10.0,
            "Z": np.arange(rows, dtype=float) * 0.2,
            "GR": np.linspace(40.0, 64.0, rows),
            "TVT_input": np.r_[
                100.0 + np.arange(known_rows, dtype=float),
                np.full(rows - known_rows, np.nan),
            ],
        }
    )
    horizontal.loc[[4, 33, 35], "GR"] = np.nan
    typewell = pd.DataFrame(
        {
            "TVT": np.linspace(80.0, 180.0, 401),
            "GR": np.linspace(35.0, 70.0, 401),
        }
    )
    return horizontal, typewell


def test_contract_is_stage0_only_and_run_is_fail_closed(train, config):
    stage0 = deepcopy(config)
    stage0["implementation"]["stage_1_implemented"] = False
    stage0["execution"]["run_stage_0"] = True
    stage0["execution"]["run_stage_1"] = False
    counts = train.validate_scientific_contract(stage0)
    assert counts == {
        "stage_0_technical_audits": 1,
        "reporting_folds": 0,
        "stage_0_hmm_well_runs": 0,
        "models": 0,
        "trained_folds": 0,
        "boosters": 0,
        "parent_control_retraining": False,
        "conditional_stage_1_hmm_well_runs": 773,
    }
    train.validate_scientific_contract(stage0, require_run_approval=True)

    denied = deepcopy(stage0)
    denied["execution"]["run_stage_0"] = False
    with pytest.raises(RuntimeError, match="package/push/run is not approved"):
        train.validate_scientific_contract(denied, require_run_approval=True)

    stage1 = deepcopy(stage0)
    stage1["execution"]["run_stage_1"] = True
    with pytest.raises(ValueError, match="execution.run_stage_1"):
        train.validate_scientific_contract(stage1, require_run_approval=True)

    changed_formula = deepcopy(stage0)
    changed_formula["model"]["missing_weight"]["missing_formula"] = "different"
    with pytest.raises(ValueError, match="missing_formula"):
        train.validate_scientific_contract(changed_formula)


def test_stage1_contract_records_exactly_one_authorized_candidate(stage1, config):
    stage1.validate_scientific_contract(config)
    authorized = deepcopy(config)
    authorized["execution"]["train_run_approved"] = True
    authorized["execution"]["run_stage_1"] = True
    authorized["execution"]["run_hmm"] = True
    stage1.validate_scientific_contract(authorized, require_run_approval=True)
    assert config["model"]["active_variants"] == ["missing_distance_half_life_8_floor_0p25"]
    assert config["execution_contract"]["scientific_variants"] == 1
    assert config["execution_contract"]["reporting_folds"] == 5
    assert config["execution_contract"]["hmm_well_runs"] == 773
    assert config["execution_contract"]["boosters"] == 0
    assert config["execution_contract"]["parent_control_retraining"] is False
    assert config["execution"]["stage_1_execution_completed"] is True
    assert config["execution"]["stage_1_promotion_passed"] is False
    assert config["execution"]["run_stage_1"] is False
    assert config["execution"]["run_hmm"] is False
    assert config["execution"]["run_inference"] is False
    assert config["execution"]["create_submission"] is False

    blocked = deepcopy(authorized)
    blocked["execution"]["run_stage_1"] = False
    with pytest.raises(RuntimeError, match="Stage 1 Kaggle package/push/run"):
        stage1.validate_scientific_contract(blocked, require_run_approval=True)


def test_stage1_weighted_gaussian_changes_only_row_emission_precision(stage1):
    observed = np.array([10.0, 20.0])
    states = np.array([5.0, 15.0, 25.0])
    base = stage1.build_weighted_gaussian_emission(observed, states, 10.0, np.ones(2), 600.0)
    weighted = stage1.build_weighted_gaussian_emission(
        observed, states, 10.0, np.array([1.0, 0.25]), 600.0
    )
    assert np.array_equal(weighted[0], base[0])
    np.testing.assert_array_equal(weighted[1], np.float32(0.25) * base[1])


def test_stage1_preserves_exp209_sigma_grid_and_missing_distance(stage1, config):
    horizontal, typewell = synthetic_well()
    prepared = stage1.prepare_hmm_inputs(horizontal, typewell, config)
    known = horizontal["TVT_input"].notna().to_numpy()
    typewell_at_known = np.interp(
        horizontal.loc[known, "TVT_input"].to_numpy(np.float64),
        typewell["TVT"].to_numpy(np.float64),
        typewell["GR"].to_numpy(np.float64),
    )
    expected_residual = (
        horizontal.loc[known, "GR"].fillna(0.0).to_numpy(np.float64) - typewell_at_known
    )
    expected_sigma = float(np.clip(np.std(expected_residual, ddof=0), 10.0, 60.0))
    assert prepared["sigma_gr"] == pytest.approx(expected_sigma)
    assert prepared["rates"].shape == (41,)
    assert prepared["rates"][0] == pytest.approx(-prepared["rates"][-1])
    assert prepared["raw_gr_observed"].tolist() == [
        True,
        True,
        True,
        False,
        True,
        False,
        True,
        True,
    ]
    assert prepared["nearest_finite_row_distance"].tolist() == [0, 0, 0, 1, 0, 1, 0, 0]
    expected_weight = np.ones(8, dtype=np.float64)
    expected_weight[[3, 5]] = np.exp2(-1.0 / 8.0)
    np.testing.assert_array_equal(prepared["confidence_weight"], expected_weight)


def test_stage1_exact_kernel_matches_parent_for_weighted_emission(stage1, config):
    parent = load_module(PARENT_SOURCE, "exp209_parent_for_exp358_stage1")
    horizontal, typewell = synthetic_well()
    prepared = stage1.prepare_hmm_inputs(horizontal, typewell, config)
    candidate = stage1.run_exact_hmm_variant(
        prepared,
        float(prepared["sigma_gr"]),
        config,
    )
    emission = stage1.build_weighted_gaussian_emission(
        prepared["observed_gr"],
        prepared["state_gr"],
        float(prepared["sigma_gr"]),
        prepared["confidence_weight"],
        600.0,
    )
    expected_post, expected_loglik = parent._hmm2_fb(
        emission,
        prepared["dm"].astype(np.float64),
        prepared["dz"].astype(np.float64),
        0.35,
        prepared["rates"].astype(np.float64),
        0.002,
        0.02,
        float(prepared["start_p"]),
        0.75,
        float(prepared["init_rate"]),
        0.01,
        1.0,
        0.998,
    )
    grid = prepared["grid"].astype(np.float64)
    expected_mean = expected_post @ grid
    expected_variance = expected_post @ (grid**2) - expected_mean**2
    expected_std = np.sqrt(np.maximum(expected_variance, 0.0))
    np.testing.assert_allclose(candidate["mean"], expected_mean, rtol=0.0, atol=1.0e-10)
    np.testing.assert_allclose(candidate["std"], expected_std, rtol=0.0, atol=1.0e-10)
    assert candidate["loglik"] == pytest.approx(expected_loglik, abs=1.0e-10)
    assert candidate["posterior_row_sum_max_abs_error"] < 1.0e-6
    assert candidate["emission_finite_coverage"] == 1.0
    assert candidate["emission_clip_z2"] == 600.0
    assert candidate["weight_application_count"] == 1


def test_stage1_promotion_gate_uses_preregistered_scopes_and_weight_guards(stage1, config):
    local = deepcopy(config)
    wells = ["a", "b", "c", "d", "e"]
    raw_patterns = {
        "a": [True] * 5,
        "b": [True] * 5,
        "c": [False, True, True, True, True],
        "d": [False, False, True, True, True],
        "e": [False] * 5,
    }
    missing_distances = iter([1, 4, 20, 1, 4, 20, 1, 4])
    rows: list[dict[str, object]] = []
    for well in wells:
        pattern = raw_patterns[well]
        missing_fraction = 1.0 - float(np.mean(pattern))
        for fold, observed in enumerate(pattern):
            if observed:
                distance = 0
                weight = 1.0
                gap_bucket = "observed"
            else:
                distance = next(missing_distances)
                weight = max(0.25, float(np.exp2(-distance / 8.0)))
                gap_bucket = (
                    "gap_1_3" if distance <= 3 else "gap_4_15" if distance <= 15 else "gap_16_plus"
                )
            rows.append(
                {
                    "well_id": well,
                    "fold": fold,
                    "true_tvt": 0.0,
                    "md_since": [100.0, 500.0, 1200.0, 1400.0, 1600.0][fold],
                    "raw_hmm_tvt": 1.0,
                    "likpf_mean": 1.0,
                    "raw_hmm_likpf_50_50": 1.0,
                    "raw_gr_observed": observed,
                    "raw_gr_missing": not observed,
                    "nearest_finite_row_distance": distance,
                    "confidence_weight": weight,
                    "missing_run_length": 0 if observed else distance,
                    "gap_bucket": gap_bucket,
                    "interpolated_gr": 50.0,
                    "evaluation_missing_fraction": missing_fraction,
                    "hidden_like_spatial": True,
                    "hidden_like_typewell_purged": True,
                    f"{stage1.VARIANT_ORDER[0]}_hmm_tvt": 0.8,
                    f"{stage1.VARIANT_ORDER[0]}_likpf_50_50": 0.9,
                }
            )
    frame = pd.DataFrame(rows)
    local["validation"].update({"expected_rows": len(frame), "expected_wells": len(wells)})
    local["execution_contract"]["hmm_well_runs"] = len(wells)
    local["references"].update(
        {
            "exp209_raw_hmm_rmse": 1.0,
            "exp209_likpf_rmse": 1.0,
            "exp209_hmm_likpf_50_50_rmse": 1.0,
        }
    )
    paired, by_well = stage1.build_paired_metrics(frame, local)
    runtime = pd.DataFrame(
        {
            "well_id": wells,
            "variant": [stage1.VARIANT_ORDER[0]] * len(wells),
            "posterior_row_sum_max_abs_error": np.zeros(len(wells)),
            "emission_clip_z2": np.full(len(wells), 600.0),
            "weight_application_count": np.ones(len(wells), dtype=int),
            "emission_finite_coverage": np.ones(len(wells)),
            "raw_observed_rows": [sum(raw_patterns[well]) for well in wells],
            "raw_missing_rows": [5 - sum(raw_patterns[well]) for well in wells],
        }
    )
    observation_audit = pd.DataFrame(
        {
            "well_id": wells,
            "exp209_zero_fill_sigma": np.full(len(wells), 30.0),
        }
    )
    preflight = {"raw_train": {"content_sha256": "a" * 64}}
    gate = stage1.evaluate_promotion_gate(
        paired,
        by_well,
        frame,
        runtime,
        observation_audit,
        preflight,
        1.0,
        local,
    )
    assert gate["passed"] is True
    assert gate["primary_direct_gate"]["folds_improved"] == 5
    assert len(gate["primary_direct_gate"]["fold_readout"]) == 5
    assert set(gate["primary_direct_gate"]["scope_readout"]) >= {
        "raw_gr_observed",
        "raw_gr_missing",
        "md_since_1000_plus",
        "hidden_like_spatial",
        "hidden_like_typewell_purged",
    }
    assert gate["primary_direct_gate"]["by_well_readout"]["wells"] == 5
    assert gate["primary_direct_gate"]["by_well_readout"]["worst_well_id"] in wells
    assert all(gate["primary_direct_gate"]["required_non_regression_scopes"].values())
    assert gate["technical_gate"]["observed_weight_exact_one"] is True
    assert gate["technical_gate"]["missing_weight_formula_exact"] is True
    assert gate["technical_gate"]["weight_applied_exactly_once"] is True

    broken_frame = frame.copy()
    broken_frame.loc[
        broken_frame["well_id"] == "e",
        f"{stage1.VARIANT_ORDER[0]}_hmm_tvt",
    ] = 2.0
    broken_frame.loc[
        broken_frame["well_id"] == "e",
        f"{stage1.VARIANT_ORDER[0]}_likpf_50_50",
    ] = 1.5
    broken_paired, broken_by_well = stage1.build_paired_metrics(broken_frame, local)
    broken = stage1.evaluate_promotion_gate(
        broken_paired,
        broken_by_well,
        broken_frame,
        runtime,
        observation_audit,
        preflight,
        1.0,
        local,
    )
    assert broken["passed"] is False
    assert broken["decision"] == "missing_distance_exp209_failed_close_without_rescue"


def test_nearest_finite_distance_handles_leading_internal_and_trailing_gaps(train):
    raw_gr = np.array([np.nan, np.nan, 10.0, np.nan, np.nan, np.nan, 20.0, np.nan])
    result = train.build_missing_distance_confidence(raw_gr)
    np.testing.assert_array_equal(
        result["nearest_finite_row_distance"],
        np.array([2, 1, 0, 1, 2, 1, 0, 1], dtype=np.int32),
    )
    expected = np.ones(len(raw_gr), dtype=np.float64)
    missing = ~np.isfinite(raw_gr)
    expected[missing] = np.maximum(
        0.25,
        np.exp2(-result["nearest_finite_row_distance"][missing].astype(np.float64) / 8.0),
    )
    np.testing.assert_array_equal(result["confidence_weight"], expected)
    assert np.array_equal(
        result["confidence_weight"][np.isfinite(raw_gr)],
        np.ones(2, dtype=np.float64),
    )


def test_all_missing_well_uses_exact_floor_and_typewell_mean(train):
    raw_gr = np.array([np.nan, np.inf, -np.inf, np.nan])
    confidence = train.build_missing_distance_confidence(raw_gr)
    assert confidence["no_finite_gr_fallback"]
    np.testing.assert_array_equal(
        confidence["nearest_finite_row_distance"],
        np.full(4, -1, dtype=np.int32),
    )
    np.testing.assert_array_equal(
        confidence["confidence_weight"],
        np.full(4, 0.25, dtype=np.float64),
    )
    interpolated = train.parent_interpolated_gr(raw_gr, np.array([20.0, 30.0]))
    np.testing.assert_array_equal(interpolated, np.full(4, 25.0))


def test_half_life_and_floor_grid_are_forbidden(train):
    with pytest.raises(ValueError, match="forbids a half-life/floor grid"):
        train.build_missing_distance_confidence(np.array([1.0, np.nan]), half_life_rows=4.0)
    with pytest.raises(ValueError, match="forbids a half-life/floor grid"):
        train.build_missing_distance_confidence(np.array([1.0, np.nan]), minimum_weight=0.1)


def test_surface_is_unknown_suffix_only_and_truth_free(train):
    horizontal = pd.DataFrame(
        {
            "GR": [10.0, np.nan, np.nan, np.nan, 20.0, np.nan, np.nan, 30.0],
            "TVT_input": [100.0, 101.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        }
    )
    surface, summary = train.build_well_weight_surface(
        "well_a", horizontal, np.array([5.0, 15.0, 25.0])
    )
    assert list(surface["row_idx"]) == [2, 3, 4, 5, 6, 7]
    assert list(surface["suffix_offset"]) == list(range(6))
    assert len(surface) == 6
    assert summary["known_prefix_rows"] == 2
    assert summary["score_rows"] == 6
    assert summary["truth_columns_read"] == 0
    assert "TVT" not in surface.columns
    assert "tvt_true" not in surface.columns
    assert np.isfinite(surface["interpolated_gr"]).all()


def test_pre_freeze_reader_never_loads_unknown_suffix_truth(train, tmp_path):
    pd.DataFrame(
        {
            "GR": [10.0, np.nan],
            "TVT_input": [100.0, np.nan],
            "TVT": [100.0, 102.0],
            "error": [0.0, 999.0],
        }
    ).to_csv(tmp_path / "well_a__horizontal_well.csv", index=False)
    safe = train.load_horizontal_without_truth("well_a", tmp_path)
    assert list(safe.columns) == ["GR", "TVT_input"]


def test_stage0_gate_requires_exact_formula_and_nontrivial_missing_weights(train, config):
    horizontal = pd.DataFrame(
        {
            "GR": [10.0, np.nan, np.nan, np.nan, 20.0, np.nan, np.nan, 30.0],
            "TVT_input": [100.0, 101.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        }
    )
    surface, summary = train.build_well_weight_surface(
        "well_a", horizontal, np.array([5.0, 15.0, 25.0])
    )
    synthetic_config = deepcopy(config)
    synthetic_config["validation"]["expected_rows"] = len(surface)
    synthetic_config["validation"]["expected_wells"] = 1
    per_well = pd.DataFrame([summary])
    freeze = {
        "frozen_before_truth_attachment": True,
        "truth_attached": False,
        "truth_columns_read": [],
    }
    gate = train.evaluate_stage0_gate(
        surface,
        per_well,
        freeze,
        synthetic_config,
        runtime_seconds=0.1,
    )
    assert gate["passed"]
    assert gate["stage_1_technical_eligibility"]
    assert not gate["stage_1_implemented"]
    assert not gate["stage_1_execution_approved"]
    assert gate["missing_weight_unique_count"] >= 2

    tampered = surface.copy()
    missing_row = tampered.index[tampered["raw_gr_missing"]][0]
    tampered.loc[missing_row, "confidence_weight"] = 0.5
    failed = train.evaluate_stage0_gate(
        tampered,
        per_well,
        freeze,
        synthetic_config,
        runtime_seconds=0.1,
    )
    assert not failed["passed"]
    assert not failed["checks"]["missing_weight_formula_exact"]


def test_inference_candidate_is_explicitly_fail_closed(inference, config):
    inference.validate_inference_disabled(config)
    with pytest.raises(RuntimeError, match="not implemented or approved"):
        inference.run_inference(config)

    enabled = deepcopy(config)
    enabled["inference"]["enabled"] = True
    with pytest.raises(ValueError, match="inference must remain disabled"):
        inference.validate_inference_disabled(enabled)


def test_compact_sources_are_not_notebook_unsafe_or_helper_entrypoints():
    train_text = TRAIN_SOURCE.read_text()
    stage1_text = STAGE1_SOURCE.read_text()
    inference_text = INFERENCE_SOURCE.read_text()
    assert "__file__" not in train_text
    assert "__file__" not in stage1_text
    assert "__file__" not in inference_text
    assert "from settings import" not in train_text
    assert "from settings import" not in stage1_text
    assert "from settings import" not in inference_text
    assert "run_stage0(CONFIG)" in train_text
    assert "def build_weighted_gaussian_emission" in stage1_text
    assert "def _hmm2_fb" in stage1_text
    assert "def generate_and_freeze_predictions" in stage1_text
    assert "def evaluate_promotion_gate" in stage1_text
    assert "SUMMARY = run_full_experiment(CONFIG)" in stage1_text
    assert "huber_log_likelihood" not in stage1_text
    assert "run_inference(CONFIG)" in inference_text
