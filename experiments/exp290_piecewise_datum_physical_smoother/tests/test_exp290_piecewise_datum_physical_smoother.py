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
EXP_DIR = ROOT / "experiments" / "exp290_piecewise_datum_physical_smoother"
TRAIN_SOURCE = EXP_DIR / "exp290_piecewise_datum_physical_smoother_compact_selfcontained_train.py"
INFERENCE_SOURCE = (
    EXP_DIR / "exp290_piecewise_datum_physical_smoother_compact_selfcontained_inference.py"
)


def load_module(path: Path = TRAIN_SOURCE, name: str = "exp290_train"):
    previous = os.environ.get("EXP290_IMPORT_ONLY")
    os.environ["EXP290_IMPORT_ONLY"] = "1"
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        if previous is None:
            os.environ.pop("EXP290_IMPORT_ONLY", None)
        else:
            os.environ["EXP290_IMPORT_ONLY"] = previous


def load_config() -> dict:
    value = yaml.safe_load((EXP_DIR / "config.yaml").read_text())
    assert isinstance(value, dict)
    return value


def synthetic_target_safe(rows: int = 1200, known_rows: int = 1000) -> pd.DataFrame:
    index = np.arange(rows, dtype=np.int64)
    tvt_input = 1000.0 + 0.25 * index
    tvt_input[known_rows:] = np.nan
    return pd.DataFrame(
        {
            "id": [f"id-{value}" for value in index],
            "row_idx": index,
            "MD": index.astype(float),
            "X": 100.0 + 0.5 * index,
            "Y": 200.0 + 0.2 * index,
            "Z": -1000.0 - 0.1 * index,
            "GR": 80.0 + 5.0 * np.sin(index / 20.0),
            "TVT_input": tvt_input,
        }
    )


def test_config_is_one_stage0_contract_and_zero_boosters() -> None:
    module = load_module(name="exp290_contract")
    config = load_config()
    module.validate_scientific_contract(config)
    assert config["experiment"]["route"] == "pf_beam"
    assert config["experiment"]["status"] == "stage0_guard_failed_branch_closed"
    assert config["experiment"]["phase"] == "stage0_scientific_guard_failed"
    assert config["stages"]["stage0"]["active_contracts"] == 1
    assert config["stages"]["stage0"]["ml_configs"] == 0
    assert config["stages"]["stage0"]["trained_folds"] == 0
    assert config["stages"]["stage0"]["boosters"] == 0
    assert config["execution"]["active_variants"] == 1
    assert config["execution"]["implementation_approved"] is True
    assert config["execution"]["kaggle_push_approved"] is True
    assert (
        config["execution"]["kaggle_push_approval_source"]
        == "user_message_run_exp290_2026_07_19"
    )
    assert config["execution"]["control_or_parent_retraining"] is False
    assert config["inference"]["enabled"] is False
    assert (
        config["stages"]["stage1"]["implementation_status"]
        == "closed_stage0_scientific_guard_failed"
    )


def test_target_safe_loader_never_materializes_truth_or_formation(tmp_path: Path) -> None:
    module = load_module(name="exp290_target_safe")
    source = synthetic_target_safe(rows=30, known_rows=20).drop(columns=["id", "row_idx"])
    source["TVT"] = 1000.0
    source["ANCC"] = -8000.0
    source["ASTNU"] = -8100.0
    path = tmp_path / "well-a__horizontal_well.csv"
    source.to_csv(path, index=False)
    loaded = module.load_target_safe_horizontal(path)
    assert tuple(loaded.columns) == module.TARGET_SAFE_COLUMNS
    assert not module.TARGET_FORBIDDEN_COLUMNS.intersection(loaded.columns)
    with pytest.raises(ValueError, match="forbidden columns"):
        module.validate_target_safe_frame(loaded.assign(TVT=1.0))


def test_fixed_pseudocuts_are_chronological_and_mask_future_truth() -> None:
    module = load_module(name="exp290_pseudocut")
    config = load_config()
    frame = synthetic_target_safe()
    cuts = []
    for horizon in (512, 256, 128):
        masked, held, manifest = module.build_fixed_pseudocut(
            "well-a", frame, horizon, config
        )
        cuts.append(manifest["cut_row"])
        assert len(held) == 128
        assert held["held_tvt"].notna().all()
        assert masked.loc[manifest["cut_row"] + 1 :, "TVT_input"].isna().all()
        assert manifest["held_tvt_access_before_prediction_freeze"] == 0
    assert cuts == [487, 743, 871]
    assert cuts == sorted(cuts)


def test_huber_affine_calibration_is_robust_and_finite() -> None:
    module = load_module(name="exp290_huber")
    template = np.linspace(20.0, 120.0, 200)
    observed = 1.4 * template - 7.0
    observed[::25] += 500.0
    scale, offset, sigma = module.huber_affine_fit(template, observed)
    assert scale == pytest.approx(1.4, abs=0.08)
    assert offset == pytest.approx(-7.0, abs=5.0)
    assert np.isfinite(sigma) and sigma >= 1.0


def test_spatial_neighbor_order_is_stable_and_excludes_self() -> None:
    module = load_module(name="exp290_neighbors")
    calibration = pd.DataFrame(
        {
            "well_id": ["a", "b", "c", "d"],
            "x_center": [0.0, 1.0, -1.0, 2.0],
            "y_center": [0.0, 0.0, 0.0, 0.0],
        }
    )
    first = module.stable_spatial_neighbors(
        calibration, 0.0, 0.0, 3, exclude_well="a"
    )
    second = module.stable_spatial_neighbors(
        calibration, 0.0, 0.0, 3, exclude_well="a"
    )
    pd.testing.assert_frame_equal(first, second)
    assert "a" not in first["well_id"].tolist()
    assert first["well_id"].tolist()[:2] == ["b", "c"]


def test_transition_is_normalized_locked_and_non_cumulative() -> None:
    module = load_module(name="exp290_transition")
    states = np.asarray([-1.0, 0.0, 1.0])
    phase_count = 5
    transition = module.build_transition_log_matrix(
        states, phase_count, hazard=0.2, jump_scale=1.0
    )
    probability = np.exp(transition)
    np.testing.assert_allclose(probability.sum(axis=1), 1.0, atol=1e-12)
    zero_phase0 = 1 * phase_count
    assert probability[zero_phase0, zero_phase0 + 1] == pytest.approx(1.0)
    assert np.count_nonzero(probability[zero_phase0]) == 1
    zero_eligible = zero_phase0 + phase_count - 1
    assert probability[zero_eligible, zero_eligible] == pytest.approx(0.8)
    reset_destinations = np.flatnonzero(probability[zero_eligible] > 0)
    assert all(
        destination == zero_eligible or destination % phase_count == 0
        for destination in reset_destinations
    )


def test_exact_posterior_mean_is_finite_bounded_and_not_viterbi() -> None:
    module = load_module(name="exp290_solver")
    config = load_config()
    states = module.state_grid_from_config(config)
    emission = np.vstack(
        [
            -0.5 * np.square((states - 5.0) / 2.0),
            -0.5 * np.square((states - 5.0) / 2.0),
        ]
    )
    result = module.exact_semi_markov_forward_backward(
        emission,
        [0.1],
        datum_location=0.0,
        prior_scale=10.0,
        jump_scale=4.0,
        config=config,
    )
    mean = result["posterior_mean_checkpoint"]
    assert np.isfinite(mean).all()
    assert (mean >= -15.0).all() and (mean <= 15.0).all()
    assert mean.mean() > 1.0
    assert result["posterior_state"].shape == (2, 61)
    np.testing.assert_allclose(result["posterior_state"].sum(axis=1), 1.0, atol=1e-12)


def test_truth_attachment_requires_unchanged_frozen_prediction() -> None:
    module = load_module(name="exp290_freeze")
    config = load_config()
    frame = synthetic_target_safe(rows=300, known_rows=250)
    rows = np.arange(122, 250, dtype=np.int64)
    prediction = pd.DataFrame(
        {
            "well_id": ["well-a"] * 128,
            "fold": [0] * 128,
            "horizon_rows": [128] * 128,
            "cut_row": [121] * 128,
            "row_idx": rows,
            "MD": frame.iloc[rows]["MD"].to_numpy(),
            "base_geometry": 1000.0 + 0.2 * rows,
            "posterior_mean_delta": np.zeros(128),
            "prediction": 1000.0 + 0.2 * rows,
            "posterior_entropy": np.ones(128),
            "reset_probability": np.zeros(128),
            "reliability": np.ones(128),
            "event_gate_threshold": np.full(128, 0.5),
            "truth_attached": [False] * 128,
        }
    )
    held = pd.DataFrame(
        {
            "well_id": ["well-a"] * 128,
            "horizon_rows": [128] * 128,
            "cut_row": [121] * 128,
            "row_idx": rows,
            "MD": frame.iloc[rows]["MD"].to_numpy(),
            "held_tvt": 1002.0 + 0.2 * rows,
        }
    )
    frozen = module.freeze_stage0_prediction(prediction)
    with pytest.raises(ValueError, match="frozen prediction content SHA"):
        module.attach_pseudotail_truth(
            prediction.assign(prediction=prediction["prediction"] + 1.0),
            held,
            frozen_prediction_sha=frozen,
            nll_scale=5.0,
            config=config,
        )
    scored, excess = module.attach_pseudotail_truth(
        prediction,
        held,
        frozen_prediction_sha=frozen,
        nll_scale=5.0,
        config=config,
    )
    assert scored["truth_attached"].all()
    assert np.isfinite(excess)


def test_reliability_uses_only_supplied_past_history() -> None:
    module = load_module(name="exp290_reliability")
    assert module.reliability_from_history([]) == 1.0
    assert module.reliability_from_history([0.5]) == pytest.approx(np.exp(-0.5))
    assert module.reliability_from_history([0.5, 100.0]) == pytest.approx(0.1)


def test_hyperposterior_cannot_receive_datum_mean_or_jump_sign() -> None:
    module = load_module(name="exp290_hierarchy")
    config = deepcopy(load_config())
    rows = []
    for index in range(20):
        rows.append(
            {
                "well_id": f"well-{index:02d}",
                "typewell_group": "group-a" if index < 10 else "group-b",
                "x_center": float(index),
                "y_center": float(index % 3),
                "log_prefix_noise": np.log(4.0 + index / 10),
                "log_jump_scale": np.log(4.0),
                "log_reset_hazard": np.log(1.0 / 2048.0),
                "log_gr_noise": np.log(20.0),
            }
        )
    calibration = pd.DataFrame(rows)
    posterior, neighbors = module.build_hyperposterior(
        calibration,
        well="query",
        fold=0,
        typewell_group="group-a",
        x_center=3.0,
        y_center=1.0,
        current_gr_sigma=22.0,
        config=config,
    )
    assert posterior["datum_location_ft"] == 0.0
    assert posterior["datum_mean_from_typewell_or_neighbor"] == 0.0
    assert posterior["jump_sign_from_typewell_or_neighbor"] == 0.0
    assert posterior["spatial_role"] == "variance_only"
    assert len(neighbors) == 16


def test_inference_is_fail_closed() -> None:
    module = load_module(INFERENCE_SOURCE, "exp290_inference")
    contract = module.validate_disabled_inference(load_config())
    assert contract["inference_enabled"] is False
    assert contract["create_submission"] is False
    with pytest.raises(RuntimeError, match="Stage 0 known-prefix pseudo-tail"):
        module.fail_closed()
