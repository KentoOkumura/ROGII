from __future__ import annotations

import importlib.util
import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "experiments" / "exp357_exp226_huber_emission_independent_audit"
TRAIN_SOURCE = EXP_DIR / (
    "exp357_exp226_huber_emission_independent_audit_compact_selfcontained_train.py"
)
INFERENCE_SOURCE = EXP_DIR / (
    "exp357_exp226_huber_emission_independent_audit_compact_selfcontained_inference.py"
)
PARENT_SOURCE = (
    ROOT
    / "experiments"
    / "exp281_exp226_residual_offset_exact_hmm_transition_probe"
    / "exp281_exp226_residual_offset_exact_hmm_transition_probe_train.py"
)


def load_module(path: Path, name: str):
    previous = os.environ.get("EXP357_IMPORT_ONLY")
    os.environ["EXP357_IMPORT_ONLY"] = "1"
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if previous is None:
            os.environ.pop("EXP357_IMPORT_ONLY", None)
        else:
            os.environ["EXP357_IMPORT_ONLY"] = previous


@pytest.fixture(scope="module")
def train():
    return load_module(TRAIN_SOURCE, "exp357_train_test")


@pytest.fixture(scope="module")
def config():
    value = yaml.safe_load((EXP_DIR / "config.yaml").read_text())
    assert isinstance(value, dict)
    return value


def synthetic_target_free_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    known_rows = 30
    eval_rows = 100
    rows = known_rows + eval_rows
    true_tvt = 100.0 + 0.2 * np.arange(rows, dtype=float)
    horizontal = pd.DataFrame(
        {
            "MD": np.arange(rows, dtype=float),
            "Z": -true_tvt,
            "GR": 2.0 * true_tvt,
            "TVT_input": np.r_[true_tvt[:known_rows], np.full(eval_rows, np.nan)],
        }
    )
    typewell_tvt = np.linspace(0.0, 300.0, 1201)
    typewell = pd.DataFrame({"TVT": typewell_tvt, "GR": 2.0 * typewell_tvt})
    row_idx = np.arange(known_rows, rows, dtype=np.int64)
    oof_safe = pd.DataFrame(
        {
            "well_id": "well-a",
            "row_idx": row_idx,
            "suffix_offset": np.arange(eval_rows, dtype=np.int64),
            "fold": 0,
            "tvt_geop": true_tvt[row_idx] - 10.0,
        }
    )
    truth = pd.DataFrame(
        {"well_id": "well-a", "row_idx": row_idx, "tvt_true": true_tvt[row_idx]}
    )
    return oof_safe, horizontal, typewell, truth


def synthetic_gaussian_control(train, huber: pd.DataFrame) -> pd.DataFrame:
    metadata = [
        "well_id",
        "fold",
        "block_id",
        "block_start_suffix_offset",
        "block_end_suffix_offset",
        "block_start_row_idx",
        "block_end_row_idx",
        "block_row_count",
        "md_since_min_ft",
        "md_since_max_ft",
        "md_since_mid_ft",
        "observed_gr_share",
        "shift_slot",
        "shift_ft",
    ]
    gaussian = huber[metadata].copy()
    residual_z = 2.0 * (gaussian["shift_ft"].to_numpy(np.float64) - 10.0) / 10.0
    likelihood = -0.5 * np.square(residual_z)
    gaussian["likelihood_mean"] = likelihood
    gaussian["likelihood_sum"] = likelihood * gaussian["block_row_count"].to_numpy(np.float64)
    gaussian["likelihood_rank"] = train.rank_descending(likelihood)
    return gaussian


def test_stage1_override_contract_is_fixed_and_explicitly_approved(train, config):
    train.validate_scientific_contract(config)
    assert config["experiment"]["route"] == "pf_beam"
    assert config["implementation"]["enabled"] is True
    assert config["implementation"]["canonical_notebook_adopted"] is True
    assert config["implementation"]["stage_1_implemented"] is True
    assert config["execution"]["implementation_approved"] is True
    assert config["execution"]["stage_1_override_approved"] is True
    assert config["execution"]["run_stage_0"] is False
    assert config["execution_contract"]["stage_0"] == {
        "scientific_scores": 1,
        "control_scores": 1,
        "hmm_well_runs": 0,
        "model_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
    }
    assert config["execution_contract"]["stage_1_if_pass"]["hmm_well_runs"] == 773
    assert config["execution_contract"]["stage_1_override"] == {
        "scientific_variants": 1,
        "hmm_well_runs": 773,
        "model_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
    }
    unapproved = deepcopy(config)
    unapproved["execution"]["kaggle_push_approved"] = False
    unapproved["execution"]["run_stage_1"] = False
    unapproved["runtime"]["kaggle"]["train_run_on_push"] = False
    with pytest.raises(RuntimeError, match="Stage 0 package/push/run is not approved"):
        stage0_only = deepcopy(unapproved)
        stage0_only["implementation"]["stage_1_implemented"] = False
        stage0_only["model"]["stage_1"]["implemented"] = False
        train.validate_scientific_contract(stage0_only, require_run_approval=True)
    with pytest.raises(RuntimeError, match="Stage 1 package/push/run is not approved"):
        train.validate_scientific_contract(unapproved, require_run_approval=True)

    approved = deepcopy(config)
    approved["execution"]["kaggle_push_approved"] = True
    approved["execution"]["run_stage_1"] = True
    approved["runtime"]["kaggle"]["train_run_on_push"] = True
    train.validate_scientific_contract(approved, require_run_approval=True)


def test_huber_formula_is_fixed_delta_and_piecewise(train):
    zscore = np.array([0.0, 1.0, 1.345, 3.0, 10.0])
    observed = train.huber_log_likelihood(zscore, 1.345)
    expected_loss = np.where(
        np.abs(zscore) <= 1.345,
        0.5 * np.square(zscore),
        1.345 * np.abs(zscore) - 0.5 * 1.345**2,
    )
    assert np.allclose(observed, -expected_loss)
    gaussian = -0.5 * np.square(zscore)
    assert observed[0] == 0.0
    assert observed[1] == gaussian[1]
    assert observed[-1] > gaussian[-1]
    with pytest.raises(ValueError, match="positive and finite"):
        train.huber_log_likelihood(zscore, 0.0)


def test_stage1_hmm_emission_changes_only_row_likelihood_family(train, config):
    _, horizontal, typewell, _ = synthetic_target_free_inputs()
    eval_mask = horizontal["TVT_input"].isna().to_numpy()
    true_tvt = horizontal.loc[eval_mask, "GR"].to_numpy(np.float64) / 2.0
    geop = true_tvt - 10.0
    prepared = train.prepare_huber_hmm_inputs(horizontal, typewell, geop, config)
    grid = prepared["grid"]
    expected_gr = 2.0 * (geop[0] + grid)
    observed_gr = float(horizontal.loc[eval_mask, "GR"].iloc[0])
    zscore = (observed_gr - expected_gr) / float(prepared["prefix_sigma"])
    absolute = np.abs(zscore)
    expected_loss = np.where(
        absolute <= 1.345,
        0.5 * np.square(zscore),
        1.345 * absolute - 0.5 * 1.345**2,
    )
    assert np.allclose(
        prepared["emission_ll"][0], (-expected_loss).astype(np.float32)
    )
    assert prepared["emission_finite_coverage"] == 1.0
    assert prepared["delta_grid_coverage_rows"] == len(geop)


def test_stage1_exact_kernel_matches_exp281_for_identical_emission(train):
    previous = os.environ.get("EXP281_IMPORT_ONLY")
    os.environ["EXP281_IMPORT_ONLY"] = "1"
    try:
        spec = importlib.util.spec_from_file_location("exp281_parent_test", PARENT_SOURCE)
        assert spec is not None and spec.loader is not None
        parent = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(parent)
    finally:
        if previous is None:
            os.environ.pop("EXP281_IMPORT_ONLY", None)
        else:
            os.environ["EXP281_IMPORT_ONLY"] = previous
    emission = np.array(
        [[-0.1, -0.2, -0.4], [-0.3, -0.1, -0.2], [-0.2, -0.3, -0.1]],
        dtype=np.float32,
    )
    dm = np.ones(3, dtype=np.float64)
    rates = np.array([-0.01, 0.0, 0.01], dtype=np.float64)
    args = (
        emission,
        dm,
        0.35,
        rates,
        0.002,
        0.02,
        1.0,
        0.75,
        0.0,
        0.01,
        1.0,
        0.998,
    )
    observed_posterior, observed_loglik = train._hmm2_fb_huber(*args)
    expected_posterior, expected_loglik = parent._hmm2_fb(*args)
    assert np.array_equal(observed_posterior, expected_posterior)
    assert observed_loglik == expected_loglik


def test_huber_target_free_score_selects_positive_ten_shift(train, config):
    oof_safe, horizontal, typewell, _ = synthetic_target_free_inputs()
    scores, manifest = train.score_well_huber_target_free(
        oof_safe, horizontal, typewell, config
    )
    selected = scores.loc[scores["huber_likelihood_rank"].eq(1)]
    assert len(selected) == 1
    assert selected["shift_ft"].iloc[0] == 10.0
    assert manifest["blocks"] == 1
    assert manifest["huber_delta"] == 1.345
    assert manifest["huber_score_finite_coverage"] == 1.0
    assert scores["extreme_abs_z_ge_3_count"].max() > 0
    assert "tvt_true" not in scores.columns


def test_saved_control_bundle_has_rank_parity_and_stable_circular_rotation(train, config):
    oof_safe, horizontal, typewell, _ = synthetic_target_free_inputs()
    huber, _ = train.score_well_huber_target_free(oof_safe, horizontal, typewell, config)
    gaussian = synthetic_gaussian_control(train, huber)
    first, technical = train.build_target_free_score_bundle(huber, gaussian, config)
    second, _ = train.build_target_free_score_bundle(huber, gaussian, config)
    assert technical["saved_gaussian_rank_parity"] == 1.0
    assert technical["score_finite_coverage"] == 1.0
    assert first["circular_rotation"].nunique() == 1
    assert 1 <= int(first["circular_rotation"].iloc[0]) < 13
    assert np.array_equal(
        first["huber_circular_likelihood_mean"].to_numpy(),
        second["huber_circular_likelihood_mean"].to_numpy(),
    )
    assert np.allclose(
        np.sort(first["huber_likelihood_mean"]),
        np.sort(first["huber_circular_likelihood_mean"]),
    )


def test_truth_is_attached_after_score_bundle_freeze(train, config):
    oof_safe, horizontal, typewell, truth = synthetic_target_free_inputs()
    huber, _ = train.score_well_huber_target_free(oof_safe, horizontal, typewell, config)
    gaussian = synthetic_gaussian_control(train, huber)
    bundle, _ = train.build_target_free_score_bundle(huber, gaussian, config)
    assert "tvt_true" not in bundle.columns
    readout, episodes = train.build_truth_readout(bundle, oof_safe, truth, config)
    assert episodes.empty
    assert len(readout) == 1
    row = readout.iloc[0]
    assert row["nearest_shift_ft"] == 10.0
    assert row["huber_nearest_shift_rank"] == 1
    assert row["gaussian_nearest_shift_rank"] == 1
    assert row["huber_top1_regret_rmse"] == pytest.approx(0.0)


def passing_metric_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    scopes = []
    for name in (
        "overall",
        "long_tail_1000_plus",
        "hidden_like_spatial",
        "hidden_like_typewell_purged",
        "persistent_offset",
        "extreme_abs_z_ge_3",
    ):
        scopes.append(
            {
                "scope": name,
                "blocks": 100,
                "huber_minus_gaussian_mrr": 0.02,
                "huber_minus_gaussian_top3_rate": 0.02,
                "huber_minus_gaussian_mrr_gap_vs_circular": 0.01,
                "huber_minus_gaussian_top3_gap_vs_circular": 0.01,
                "huber_minus_gaussian_top1_regret_rmse_mean": -0.5,
                "huber_likelihood_top1_margin_mean": 0.1,
                "gaussian_likelihood_top1_margin_mean": 0.2,
            }
        )
    folds = pd.DataFrame(
        {
            "fold": np.arange(5),
            "huber_minus_gaussian_mrr": [0.01, 0.01, 0.01, 0.01, -0.001],
            "huber_minus_gaussian_top3_rate": [0.01, 0.01, 0.01, 0.01, -0.001],
        }
    )
    return pd.DataFrame(scopes), folds


def test_fixed_gate_is_independent_and_fail_closed(train, config):
    scopes, folds = passing_metric_frames()
    technical = {
        "saved_gaussian_rank_parity": 1.0,
        "score_finite_coverage": 1.0,
        "row_identity_coverage": 1.0,
    }
    passing = train.evaluate_guard(technical, scopes, folds, config)
    assert passing["passed"] is True
    assert passing["stage_1_eligible"] is True
    assert "exp344_dependency_pattern_matched" not in passing

    failed_scopes = scopes.copy()
    failed_scopes.loc[
        failed_scopes["scope"].eq("overall"), "huber_minus_gaussian_mrr"
    ] = 0.005
    failed = train.evaluate_guard(technical, failed_scopes, folds, config)
    assert failed["passed"] is False
    assert failed["stage_1_eligible"] is False
    assert failed["decision"] == "stage_0_failed_close_without_rescue"


def test_stage1_guard_requires_gain_fold_scope_and_tail_safety(train, config):
    parent_rmse = float(config["data"]["exp281_control"]["expected_parent_rmse"])
    exp226_rmse = float(config["data"]["exp281_control"]["expected_exp226_rmse"])
    candidate_metrics = pd.DataFrame(
        {
            "candidate": [
                "gaussian_residual_offset_hmm",
                "huber_residual_offset_hmm",
                "exp226_pred",
            ],
            "rmse": [parent_rmse, 9.30, exp226_rmse],
        }
    )
    fold_rows = []
    for fold in range(5):
        fold_rows.extend(
            [
                {
                    "fold": fold,
                    "candidate": "gaussian_residual_offset_hmm",
                    "rmse": 10.0,
                },
                {
                    "fold": fold,
                    "candidate": "huber_residual_offset_hmm",
                    "rmse": 9.9,
                },
            ]
        )
    scope_metrics = pd.DataFrame(
        {
            "scope": [
                "long_tail_1000_plus",
                "hidden_like_spatial",
                "hidden_like_typewell_purged",
            ],
            "huber_minus_gaussian_rmse": [-0.1, -0.2, -0.3],
        }
    )
    by_well = pd.DataFrame(
        {
            "well": ["a", "b", "c"],
            "delta_rmse_vs_exp281": [-0.2, -0.1, -0.05],
        }
    )
    passing = train.evaluate_stage_1_guard(
        candidate_metrics,
        pd.DataFrame(fold_rows),
        scope_metrics,
        by_well,
        finite_coverage=1.0,
        row_identity_coverage=1.0,
        config=config,
    )
    assert passing["passed"] is True
    assert passing["direct_promotion_passed"] is True
    assert passing["stage_0_prerequisite_passed"] is False

    candidate_metrics.loc[
        candidate_metrics["candidate"].eq("huber_residual_offset_hmm"), "rmse"
    ] = parent_rmse - 0.01
    failed = train.evaluate_stage_1_guard(
        candidate_metrics,
        pd.DataFrame(fold_rows),
        scope_metrics,
        by_well,
        finite_coverage=1.0,
        row_identity_coverage=1.0,
        config=config,
    )
    assert failed["passed"] is False
    assert failed["decision"] == "stage_1_failed_close_without_rescue"


def test_inference_and_sources_are_fail_closed_and_self_contained(train, config):
    inference = load_module(INFERENCE_SOURCE, "exp357_inference_test")
    contract = inference.validate_disabled_inference(config)
    assert contract["scientific_scores"] == 1
    assert contract["control_scores"] == 1
    assert contract["model_configs"] == 0
    assert contract["hmm_well_runs"] == 0
    assert contract["stage_1_hmm_well_runs"] == 773
    assert contract["stage_1_implemented"] is True
    with pytest.raises(RuntimeError, match="raw-test inference and submission"):
        inference.stop_disabled_inference(config)

    train_text = TRAIN_SOURCE.read_text()
    inference_text = INFERENCE_SOURCE.read_text()
    assert "def huber_log_likelihood" in train_text
    assert "def score_well_huber_target_free" in train_text
    assert "def build_target_free_score_bundle" in train_text
    assert "def evaluate_guard" in train_text
    assert "def run_stage_0_experiment" in train_text
    assert "def prepare_huber_hmm_inputs" in train_text
    assert "def _hmm2_fb_huber" in train_text
    assert "def run_stage_1_experiment" in train_text
    assert "from settings import" not in train_text
    assert "from src" not in train_text
    assert "Path(__file__)" not in train_text
    assert "Path(__file__)" not in inference_text
