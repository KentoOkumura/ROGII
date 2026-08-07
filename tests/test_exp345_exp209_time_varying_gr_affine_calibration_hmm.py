from __future__ import annotations

import importlib.util
import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "experiments" / "exp345_exp209_time_varying_gr_affine_calibration_hmm"
TRAIN_SOURCE = EXP_DIR / (
    "exp345_exp209_time_varying_gr_affine_calibration_hmm_compact_selfcontained_train.py"
)
PARENT_SOURCE = (
    ROOT
    / "experiments"
    / "exp209_exp072_exp205_joint_exact_parity_fast_cache_generation"
    / "exact_hmm_smoother.py"
)


def load_module(path: Path, name: str):
    previous = os.environ.get("EXP345_IMPORT_ONLY")
    os.environ["EXP345_IMPORT_ONLY"] = "1"
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if previous is None:
            os.environ.pop("EXP345_IMPORT_ONLY", None)
        else:
            os.environ["EXP345_IMPORT_ONLY"] = previous


def load_config(module):
    return module.read_yaml(EXP_DIR / "config.yaml")


def synthetic_frames(rows: int = 1100) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    md = 1000.0 + np.arange(rows, dtype=float)
    tvt = 100.0 + 0.1 * np.arange(rows, dtype=float)
    typewell_tvt = np.linspace(50.0, 260.0, 1000)
    typewell_gr = 50.0 + 20.0 * np.sin(typewell_tvt / 12.0)
    raw_gr = 1.2 * np.interp(tvt, typewell_tvt, typewell_gr) + 5.0
    horizontal = pd.DataFrame(
        {
            "MD": md,
            "Z": np.zeros(rows, dtype=float),
            "GR": raw_gr,
            "TVT_input": tvt,
        }
    )
    typewell = pd.DataFrame({"TVT": typewell_tvt, "GR": typewell_gr})
    return horizontal, typewell, tvt


def test_contract_is_fail_closed_after_stage0_scientific_gate_failure() -> None:
    module = load_module(TRAIN_SOURCE, "exp345_contract")
    config = load_config(module)
    module.validate_scientific_contract(config)
    assert module.get_nested(config, "implementation.enabled") is True
    assert module.get_nested(config, "implementation.canonical_notebook_adopted") is True
    assert module.get_nested(config, "execution.implementation_approved") is True
    assert module.get_nested(config, "execution.kaggle_push_approved") is False
    assert module.get_nested(config, "execution.run_microbenchmark") is False
    assert module.get_nested(config, "execution.run_stage_0") is False
    assert module.get_nested(config, "execution.run_stage_1") is False
    assert module.get_nested(config, "execution.runtime_gate_passed") is True
    assert module.get_nested(config, "execution.runtime_gate_evidence_sha256") == (
        "744e545edeca0864ae1b595bc30062457bf742083bcab908484e1fb53b32aca1"
    )
    assert module.get_nested(config, "execution.stage_0_gate_evidence_sha256") == (
        "39296d1b900463c27f1fd65fbaa265e3c1a3a6b9d42afd9322eb03ac6140525a"
    )
    assert module.get_nested(config, "execution.stage_0_gate_passed") is False
    assert module.get_nested(config, "execution.stage_1_eligible") is False
    assert module.get_nested(config, "inference.enabled") is False
    with pytest.raises(RuntimeError, match="Kaggle package/push/run is not approved"):
        module.validate_scientific_contract(config, require_run_approval=True)


def test_stage_evidence_guards_are_fail_closed() -> None:
    module = load_module(TRAIN_SOURCE, "exp345_stage_guards")
    base = load_config(module)
    micro = deepcopy(base)
    micro["execution"].update(
        {
            "kaggle_push_approved": True,
            "run_microbenchmark": True,
            "run_stage_0": False,
            "run_stage_1": False,
        }
    )
    module.validate_scientific_contract(micro, require_run_approval=True)
    assert module.selected_stage(micro) == "stage_0_microbenchmark"

    stage0 = deepcopy(base)
    stage0["execution"].update(
        {
            "kaggle_push_approved": True,
            "run_microbenchmark": False,
            "run_stage_0": True,
            "run_stage_1": False,
            "runtime_gate_evidence_sha256": None,
        }
    )
    with pytest.raises(RuntimeError, match="runtime-gate evidence"):
        module.validate_scientific_contract(stage0, require_run_approval=True)

    stage0_with_evidence = deepcopy(base)
    stage0_with_evidence["execution"].update(
        {
            "kaggle_push_approved": True,
            "run_microbenchmark": False,
            "run_stage_0": True,
            "run_stage_1": False,
        }
    )
    module.validate_scientific_contract(stage0_with_evidence, require_run_approval=True)
    assert module.selected_stage(stage0_with_evidence) == "stage_0_full"

    stage1 = deepcopy(base)
    stage1["execution"].update(
        {
            "kaggle_push_approved": True,
            "run_microbenchmark": False,
            "run_stage_0": False,
            "run_stage_1": True,
        }
    )
    with pytest.raises(RuntimeError, match="Stage 0 gate PASS"):
        module.validate_scientific_contract(stage1, require_run_approval=True)


def test_robust_affine_fit_recovers_prefix_and_has_identity_fallback() -> None:
    module = load_module(TRAIN_SOURCE, "exp345_affine_fit")
    config = load_config(module)
    x = np.linspace(20.0, 100.0, 200)
    y = 1.4 * x - 7.0
    y[-1] = 10000.0
    fit = module.robust_affine_fit(x, y, config)
    assert fit["valid"] is True
    assert fit["scale_a"] == pytest.approx(1.4, abs=1.0e-10)
    assert fit["intercept_b"] == pytest.approx(-7.0, abs=1.0e-9)
    assert np.isfinite(fit["covariance"]).all()

    fallback = module.robust_affine_fit(
        np.ones(50, dtype=float),
        np.arange(50, dtype=float),
        config,
    )
    assert fallback["valid"] is False
    assert fallback["fallback_reason"] == "insufficient_typewell_gr_std"
    assert fallback["scale_a"] == 1.0
    assert fallback["intercept_b"] == 0.0


def test_stage0_mask_removes_last_640_prefix_rows_before_affine_fit() -> None:
    module = load_module(TRAIN_SOURCE, "exp345_mask")
    config = load_config(module)
    horizontal, typewell, _ = synthetic_frames()
    masked, manifest = module.stage0_masked_horizontal(horizontal, config)
    assert manifest == {
        "score_start_row": 460,
        "score_stop_row_exclusive": 1100,
        "score_rows": 640,
        "visible_prefix_rows": 460,
        "original_known_prefix_rows": 1100,
        "truncated_at_original_prefix_boundary": True,
    }
    assert masked.loc[:459, "TVT_input"].notna().all()
    assert masked.loc[460:, "TVT_input"].isna().all()
    raw = module.prefix_process_noise_raw(masked, typewell, config)
    assert raw["state_fits"] > 1
    assert raw["process_increments"] == raw["state_fits"] - 1


def test_causal_schedule_skips_missing_raw_gr_and_freezes_finite_state() -> None:
    module = load_module(TRAIN_SOURCE, "exp345_schedule")
    config = load_config(module)
    horizontal, typewell, true_tvt = synthetic_frames()
    masked, _ = module.stage0_masked_horizontal(horizontal, config)
    prepared = module.prepare_hmm_inputs(masked, typewell, config)
    eval_index = prepared["eval_index"]
    base_mean = true_tvt[eval_index]
    base_std = np.ones(len(eval_index), dtype=float)
    masked.loc[eval_index[:5], "GR"] = np.nan
    prepared_missing = module.prepare_hmm_inputs(masked, typewell, config)
    schedule, audit = module.causal_affine_schedule(
        masked,
        typewell,
        prepared_missing,
        base_mean,
        base_std,
        {"q_intercept": 1.0e-6, "q_log_scale": 1.0e-10},
        config,
    )
    assert not schedule["raw_gr_update"].iloc[:5].any()
    assert schedule["raw_gr_update"].iloc[5:].all()
    assert audit["fallback"] is False
    assert audit["finite_updates"] == len(eval_index) - 5
    assert schedule["affine_scale_a"].between(0.25, 4.0).all()
    assert np.isfinite(schedule[["affine_scale_a", "affine_intercept_b"]].to_numpy(float)).all()


def test_identity_schedule_is_exactly_the_exp209_parent_hmm() -> None:
    module = load_module(TRAIN_SOURCE, "exp345_hmm")
    parent = load_module(PARENT_SOURCE, "exp345_parent_hmm")
    config = load_config(module)
    rows = 36
    known_rows = 30
    horizontal = pd.DataFrame(
        {
            "MD": np.arange(rows, dtype=float) * 10.0,
            "Z": np.arange(rows, dtype=float) * 0.2,
            "GR": np.linspace(40.0, 62.0, rows),
            "TVT_input": np.r_[
                100.0 + np.arange(known_rows, dtype=float),
                [np.nan] * (rows - known_rows),
            ],
        }
    )
    horizontal.loc[[4, 32, 33], "GR"] = np.nan
    typewell = pd.DataFrame(
        {
            "TVT": np.linspace(80.0, 170.0, 181),
            "GR": np.linspace(35.0, 67.0, 181),
        }
    )
    prepared = module.prepare_hmm_inputs(horizontal, typewell, config)
    count = len(prepared["eval_index"])
    candidate = module.run_exact_hmm(
        prepared,
        config,
        scale_schedule=np.ones(count, dtype=float),
        intercept_schedule=np.zeros(count, dtype=float),
    )
    parent_result = parent.run_hmm2(
        horizontal,
        typewell,
        step=0.35,
        n_rates=41,
        rate_span=0.10,
        sig_r=0.002,
        sig_p=0.02,
        emission="gauss",
        lam=1.0,
        sigma_mode="std",
        start_sig=0.75,
        r0_sig=0.01,
        band_pad=100.0,
        mom=0.998,
        rate_center="zero",
        return_post=True,
    )
    np.testing.assert_allclose(
        candidate["mean"], parent_result["mean_eval"], rtol=0.0, atol=1.0e-10
    )
    np.testing.assert_allclose(candidate["std"], parent_result["std_eval"], rtol=0.0, atol=1.0e-10)
    assert prepared["prefix_scale"]["sigma_gr"] == pytest.approx(parent_result["prefix_sigma"])
    assert candidate["posterior_row_sum_max_abs_error"] < 1.0e-6


def test_truth_free_horizontal_loader_drops_tvt_and_error_columns(
    tmp_path: Path,
) -> None:
    module = load_module(TRAIN_SOURCE, "exp345_truth_boundary")
    pd.DataFrame(
        {
            "MD": [0.0, 1.0],
            "Z": [0.0, 0.1],
            "GR": [50.0, np.nan],
            "TVT_input": [100.0, np.nan],
            "TVT": [100.0, 101.0],
            "error": [0.0, 99.0],
        }
    ).to_csv(tmp_path / "a__horizontal_well.csv", index=False)
    frame = module.load_horizontal_without_truth("a", tmp_path)
    assert list(frame.columns) == ["MD", "Z", "GR", "TVT_input"]
