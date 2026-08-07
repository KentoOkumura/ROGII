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
EXP_DIR = ROOT / "experiments" / "exp343_acf_effective_sample_likelihood_tempering_audit"
TRAIN_SOURCE = EXP_DIR / (
    "exp343_acf_effective_sample_likelihood_tempering_audit_"
    "compact_selfcontained_train.py"
)
INFERENCE_SOURCE = EXP_DIR / (
    "exp343_acf_effective_sample_likelihood_tempering_audit_"
    "compact_selfcontained_inference.py"
)


def load_module(path: Path, name: str):
    previous = os.environ.get("EXP343_IMPORT_ONLY")
    os.environ["EXP343_IMPORT_ONLY"] = "1"
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if previous is None:
            os.environ.pop("EXP343_IMPORT_ONLY", None)
        else:
            os.environ["EXP343_IMPORT_ONLY"] = previous


@pytest.fixture(scope="module")
def train():
    return load_module(TRAIN_SOURCE, "exp343_train_test")


@pytest.fixture(scope="module")
def config():
    value = yaml.safe_load((EXP_DIR / "config.yaml").read_text())
    assert isinstance(value, dict)
    return value


def test_stage0_kaggle_v1_failed_closed_and_all_execution_is_blocked(train, config):
    contract = train.validate_scientific_contract(config)
    assert contract == {
        "diagnostic_variants": 1,
        "hmm_well_runs": 0,
        "model_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
    }
    assert config["experiment"]["status"] == "stage_0_completed_guard_failed_closed"
    assert config["implementation"]["enabled"] is True
    assert config["implementation"]["stage_1_implemented"] is False
    assert config["execution"]["kaggle_push_approved"] is False
    assert config["execution"]["run_stage_0"] is False
    assert config["runtime"]["kaggle"]["train_run_on_push"] is False
    assert config["execution"]["run_stage_1"] is False
    with pytest.raises(RuntimeError, match="package/push/run is not approved"):
        train.validate_scientific_contract(config, require_run_approval=True)

    rerun = deepcopy(config)
    rerun["execution"]["kaggle_push_approved"] = True
    rerun["execution"]["run_stage_0"] = True
    rerun["runtime"]["kaggle"]["train_run_on_push"] = True
    with pytest.raises(RuntimeError, match="first full Stage 0 run must execute"):
        train.validate_scientific_contract(rerun, require_run_approval=True)

    changed = deepcopy(config)
    changed["model"]["acf"]["rho_estimator"] = "global_biased_acf"
    with pytest.raises(ValueError, match="fixed ACF contract changed"):
        train.validate_scientific_contract(changed)


def test_known_residuals_use_raw_finite_rows_and_tail_raw_row_window(train):
    rows = 700
    tvt = 100.0 + 0.25 * np.arange(rows)
    gr = 2.0 * tvt + np.sin(np.arange(rows) / 5.0)
    gr[200] = np.nan
    tvt_input = tvt.copy()
    tvt_input[600:] = np.nan
    horizontal = pd.DataFrame({"GR": gr, "TVT_input": tvt_input})
    typewell_tvt = np.linspace(0.0, 400.0, 1601)
    typewell = pd.DataFrame({"TVT": typewell_tvt, "GR": 2.0 * typewell_tvt})

    residuals = train.build_known_residuals(
        horizontal,
        typewell,
        well_id="well-a",
        fold=2,
        tail_rows=512,
    )
    assert len(residuals) == 599
    assert residuals["known_rows"].nunique() == 1
    assert int(residuals["known_rows"].iloc[0]) == 600
    assert 200 not in residuals["row_idx"].tolist()
    assert int(residuals["in_last_512"].sum()) == 511
    assert residuals["full_run_id"].nunique() == 2
    tail = residuals.loc[residuals["in_last_512"]]
    assert int(tail["row_idx"].min()) == 88
    assert np.allclose(
        residuals["residual"],
        np.sin(residuals["row_idx"].to_numpy() / 5.0),
    )


def test_acf_pairs_do_not_cross_missing_boundary_and_tau_formula_is_fixed(train):
    first_rows = np.arange(100, dtype=np.int64)
    second_rows = np.arange(101, 201, dtype=np.int64)
    row_idx = np.r_[first_rows, second_rows]
    values = np.sin(row_idx / 9.0) + 0.1 * np.cos(row_idx / 3.0)
    runs = train.assign_contiguous_runs(row_idx)
    residuals = pd.DataFrame(
        {
            "well_id": "well-a",
            "fold": 0,
            "row_idx": row_idx,
            "in_last_512": True,
            "residual": values,
            "full_run_id": runs,
            "last_512_run_id": runs,
        }
    )
    lag_frame, metadata = train.estimate_window_acf(
        residuals,
        window="full_known_prefix",
        lags=range(1, 21),
        minimum_finite_residuals=128,
        minimum_pairs_each_lag=20,
    )
    assert int(lag_frame.loc[lag_frame["lag"].eq(1), "pair_count"].iloc[0]) == 198
    assert int(lag_frame.loc[lag_frame["lag"].eq(20), "pair_count"].iloc[0]) == 160
    assert metadata["evaluable"] is True
    expected_tau = 1.0 + 2.0 * np.maximum(lag_frame["rho"].to_numpy(), 0.0).sum()
    assert metadata["tau_raw"] == pytest.approx(expected_tau)


def test_zero_finite_residual_well_is_retained_as_fallback_candidate(train, config):
    empty = pd.DataFrame(
        columns=[
            "well_id",
            "fold",
            "row_idx",
            "in_last_512",
            "residual",
            "full_run_id",
            "last_512_run_id",
        ]
    )
    lag_frame, tau_frame = train.estimate_well_acf(
        empty,
        config,
        well_id="well-empty",
        fold=3,
    )
    assert len(lag_frame) == 40
    assert tau_frame["evaluable"].eq(False).all()
    assert tau_frame["finite_residual_count"].eq(0).all()
    assert tau_frame["tau_raw"].isna().all()
    assert tau_frame["fallback_reason"].str.contains("finite_residual_count").all()


def synthetic_raw_tau() -> pd.DataFrame:
    rows = []
    windows = ("full_known_prefix", "last_512_known_prefix_rows")
    for window_index, window in enumerate(windows):
        for fold in range(5):
            for slot in range(2):
                evaluable = not (window_index == 1 and fold == 0 and slot == 0)
                rows.append(
                    {
                        "well_id": f"w{fold}-{slot}",
                        "fold": fold,
                        "window": window,
                        "finite_residual_count": 300 + slot,
                        "contiguous_run_count": 1,
                        "minimum_pair_count": 250,
                        "lag_1_rho": 0.7,
                        "evaluable": evaluable,
                        "fallback_reason": "" if evaluable else "finite_residual_count",
                        "tau_raw": 2.0 + 0.1 * fold + 0.05 * slot if evaluable else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def test_outer_train_prior_excludes_validation_fold_and_fallback_is_prior(train, config):
    schedule, priors = train.attach_outer_train_tau_prior(synthetic_raw_tau(), config)
    expected = np.median(
        [
            2.0 + 0.1 * fold + 0.05 * slot
            for fold in range(1, 5)
            for slot in range(2)
        ]
    )
    observed = priors.loc[
        priors["window"].eq("full_known_prefix") & priors["fold"].eq(0),
        "tau_fold_median",
    ].iloc[0]
    assert observed == pytest.approx(expected)
    fallback = schedule.loc[
        schedule["window"].eq("last_512_known_prefix_rows")
        & schedule["well_id"].eq("w0-0")
    ].iloc[0]
    assert bool(fallback["fallback"])
    assert fallback["alpha"] == 0.0
    assert fallback["tau_shrunk"] == pytest.approx(fallback["tau_fold_median"])
    assert fallback["tau_eff"] == pytest.approx(fallback["tau_fold_median"])
    regular = schedule.loc[
        schedule["window"].eq("full_known_prefix") & schedule["well_id"].eq("w1-1")
    ].iloc[0]
    expected_alpha = 301.0 / 501.0
    expected_shrunk = np.exp(
        expected_alpha * np.log(regular["tau_raw"])
        + (1.0 - expected_alpha) * np.log(regular["tau_fold_median"])
    )
    assert regular["alpha"] == pytest.approx(expected_alpha)
    assert regular["tau_shrunk"] == pytest.approx(expected_shrunk)


def test_gate_is_conservative_across_both_windows(train, config):
    pooled = {
        "expected_wells": 773,
        "actual_wells": 773,
        "joint_evaluable_wells": 750,
        "joint_evaluable_fraction": 750 / 773,
        "fallback_wells": 23,
        "fallback_fraction": 23 / 773,
        "spearman_full_vs_tail": 0.8,
        "median_absolute_log_ratio": 0.1,
        "stable_folds": 4,
        "windows": {
            "full": {
                "median_tau_eff": 1.5,
                "upper_clip_fraction": 0.10,
                "fold_median_tau_max_min_ratio": 1.2,
            },
            "tail": {
                "median_tau_eff": 1.4,
                "upper_clip_fraction": 0.20,
                "fold_median_tau_max_min_ratio": 1.3,
            },
        },
    }
    fold_metrics = pd.DataFrame({"fold": np.arange(5)})
    passed = train.evaluate_stage_0_gate(pooled, fold_metrics, config)
    assert passed["passed"] is True
    assert passed["stage_1_eligible"] is True
    assert passed["decision"] == (
        "stage_0_passed_stage_1_requires_separate_user_approval"
    )

    failed_pooled = deepcopy(pooled)
    failed_pooled["windows"]["tail"]["upper_clip_fraction"] = 0.26
    failed = train.evaluate_stage_0_gate(failed_pooled, fold_metrics, config)
    assert failed["passed"] is False
    assert failed["stage_1_eligible"] is False
    assert failed["decision"] == "stage_0_failed_close_without_rescue"
    assert failed["rescue_grid_allowed"] is False


def test_inference_and_sources_are_fail_closed_and_self_contained(train, config):
    inference = load_module(INFERENCE_SOURCE, "exp343_inference_test")
    contract = inference.validate_disabled_inference(config)
    assert contract["diagnostic_variants"] == 1
    assert contract["hmm_well_runs"] == 0
    assert contract["boosters"] == 0
    with pytest.raises(RuntimeError, match="Stage 1, inference, and submission"):
        inference.stop_disabled_inference(config)
    train_text = TRAIN_SOURCE.read_text()
    inference_text = INFERENCE_SOURCE.read_text()
    assert "def build_known_residuals" in train_text
    assert "def estimate_window_acf" in train_text
    assert "def attach_outer_train_tau_prior" in train_text
    assert "def evaluate_stage_0_gate" in train_text
    assert "def run_stage_0_experiment" in train_text
    assert "from settings import" not in train_text
    assert "from src" not in train_text
    assert "Path(__file__)" not in train_text
    assert "Path(__file__)" not in inference_text
