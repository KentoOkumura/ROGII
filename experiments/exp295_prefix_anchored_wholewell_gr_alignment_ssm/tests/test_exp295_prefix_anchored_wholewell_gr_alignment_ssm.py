from __future__ import annotations

import copy
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = ROOT / "experiments" / "exp295_prefix_anchored_wholewell_gr_alignment_ssm"
TRAIN_SOURCE = EXP_DIR / (
    "exp295_prefix_anchored_wholewell_gr_alignment_ssm_compact_selfcontained_train.py"
)
INFERENCE_SOURCE = EXP_DIR / (
    "exp295_prefix_anchored_wholewell_gr_alignment_ssm_compact_selfcontained_inference.py"
)
os.environ["EXP295_IMPORT_ONLY"] = "1"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


EXP295 = load_module(TRAIN_SOURCE, "exp295_train_contract")


def load_config() -> dict:
    return yaml.safe_load((EXP_DIR / "config.yaml").read_text())


def synthetic_input(rows: int = 160, prefix_rows: int = 80):
    index = np.arange(rows, dtype=np.float64)
    tvt = 1000.0 + 0.15 * index
    tvt_input = tvt.copy()
    tvt_input[prefix_rows:] = np.nan
    type_tvt = np.arange(970.0, 1050.0, 0.5)
    type_gr = 70.0 + 9.0 * np.sin(type_tvt / 4.0)
    gr = np.interp(tvt, type_tvt, type_gr) + 1.5
    return EXP295.WellInput(
        well="well-a",
        md=10_000.0 + index,
        x=100.0 + index,
        y=200.0 + 0.2 * index,
        z=-8000.0 - 0.85 * index,
        gr=gr,
        tvt_input=tvt_input,
        typewell_tvt=type_tvt,
        typewell_gr=type_gr,
        horizontal_path=Path("well-a__horizontal_well.csv"),
        typewell_path=Path("well-a__typewell.csv"),
    ), tvt


def test_stage_a_scientific_and_gpu_cost_contract_is_fixed() -> None:
    config = load_config()
    evidence = EXP295.validate_scientific_contract(config)
    cost = EXP295.validate_stage_a_cost_contract(config)
    assert evidence["controls"] == 3
    assert cost == {
        "fold_indices": [0],
        "active_architectures": 1,
        "seeds": [42],
        "neural_model_count": 1,
        "lightgbm_config_count": 0,
        "total_boosters": 0,
        "control_model_training": 0,
        "pf_beam_well_runs": 0,
        "parent_control_retraining": False,
    }
    assert config["execution"]["implementation"] is True
    assert config["execution"]["implementation_approved"] is True
    assert config["execution"]["kaggle_push_approved"] is True
    assert config["execution"]["kaggle_push_approval_scope"] == (
        "user_message_execute_exp295_stage_a_fold0_only"
    )
    assert config["execution"]["canonical_train_notebook_adopted"] is True
    assert config["execution"]["canonical_inference_notebook_adopted"] is False
    assert config["execution"]["inference_approved"] is False
    assert config["execution"]["submission_approved"] is False
    assert config["model"]["training"]["objective"] == {
        "structured_label_nll_weight": 1.0,
        "label_observation_distribution": "gaussian",
        "label_observation_sigma_ft": 0.35,
        "local_true_state_ce_weight": 0.25,
    }

    unsafe = copy.deepcopy(config)
    unsafe["model"]["state_space"]["band_pad"] = 120.0
    with pytest.raises(ValueError, match="scientific contract changed"):
        EXP295.validate_scientific_contract(unsafe)


def test_groupkfold_identity_matches_exp202_sorted_well_contract() -> None:
    wells = [f"well-{index:03d}" for index in range(23)]
    actual = EXP295.build_fold_map(wells, 5).sort_values("well").reset_index(drop=True)
    from sklearn.model_selection import GroupKFold

    ordered = sorted(wells)
    expected_rows = []
    groups = np.asarray(ordered)
    dummy = np.zeros((len(ordered), 1), dtype=np.float32)
    for fold, (_, valid) in enumerate(GroupKFold(5).split(dummy, groups=groups)):
        expected_rows.extend({"well": ordered[index], "fold": fold} for index in valid)
    expected = pd.DataFrame(expected_rows).sort_values("well").reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, expected)


def test_pseudo_cut_masks_every_row_after_fixed_cut() -> None:
    item, _ = synthetic_input()
    official_end = EXP295.prefix_end_index(item.tvt_input)
    assert official_end == 79
    view, cut_end = EXP295.make_view_tvt_input(item.tvt_input, 32)
    assert cut_end == 47
    assert np.isfinite(view[:48]).all()
    assert np.isnan(view[48:]).all()
    with pytest.raises(ValueError, match="fewer than 32"):
        EXP295.make_view_tvt_input(item.tvt_input, 64)


def test_mask_first_loader_never_selects_truth_or_candidate_columns(tmp_path: Path) -> None:
    well = "abcd1234"
    horizontal = pd.DataFrame(
        {
            "MD": np.arange(40, dtype=float),
            "X": 1.0,
            "Y": 2.0,
            "Z": -np.arange(40, dtype=float),
            "GR": 70.0,
            "TVT_input": [*np.arange(32, dtype=float), *([np.nan] * 8)],
            "TVT": np.arange(40, dtype=float) + 1000.0,
            "candidate_leak": np.arange(40),
        }
    )
    typewell = pd.DataFrame(
        {"TVT": np.arange(40, dtype=float) + 990.0, "GR": np.arange(40, dtype=float)}
    )
    horizontal.to_csv(tmp_path / f"{well}__horizontal_well.csv", index=False)
    typewell.to_csv(tmp_path / f"{well}__typewell.csv", index=False)
    loaded = EXP295.load_well_input(well, tmp_path)
    assert not hasattr(loaded, "tvt")
    assert not hasattr(loaded, "candidate_leak")
    assert loaded.tvt_input.shape == (40,)
    truth = EXP295.load_well_truth(well, tmp_path)
    np.testing.assert_allclose(truth.tvt, horizontal["TVT"])


def test_robust_preprocessing_and_typewell_interpolation_forbid_extrapolation() -> None:
    values = np.array([1.0, 2.0, np.nan, 4.0])
    normalized, missing, center, scale = EXP295.robust_normalize(values)
    assert normalized[2] == 0.0
    assert missing.tolist() == [0.0, 0.0, 1.0, 0.0]
    assert center == 2.0
    assert scale >= 1.0

    interpolated, inside = EXP295.interpolate_no_extrapolation(
        np.array([-1.0, 0.5, 2.0]), np.array([0.0, 1.0]), np.array([10.0, 20.0])
    )
    assert inside.tolist() == [False, True, False]
    assert interpolated.tolist() == [0.0, 15.0, 0.0]


def test_huber_prefix_summary_has_neutral_fallback_and_recovers_affine_relation() -> None:
    x = np.linspace(20.0, 120.0, 64)
    y = 1.2 * x + 7.0
    summary, count = EXP295.huber_affine_summary(x, y)
    assert count == 64
    assert summary[-1] == 1.0
    assert summary[0] == pytest.approx(1.2, abs=1e-5)

    neutral, count = EXP295.huber_affine_summary(x[:20], y[:20])
    assert count == 20
    np.testing.assert_array_equal(neutral, np.zeros(5, dtype=np.float32))


def test_prepared_view_uses_only_one_well_and_fixed_exp209_grid() -> None:
    item, _ = synthetic_input()
    config = load_config()
    real = EXP295.prepare_view(item, item.tvt_input, config, view_name="official")
    shuffled = EXP295.prepare_view(
        item,
        item.tvt_input,
        config,
        view_name="shuffle",
        typewell_control="shuffle",
    )
    assert real.horizontal_channels.shape == (3, len(item.md))
    assert real.typewell_channels.shape[0] == 3
    assert real.state.rates.shape == (41,)
    assert np.diff(real.state.grid).mean() == pytest.approx(0.35)
    assert real.state.prefix_end == 79
    assert real.state.suffix_index[0] == 80
    assert real.prefix_pair_count >= 32
    assert not np.array_equal(real.typewell_channels, shuffled.typewell_channels)
    first, roll1 = EXP295.stable_circular_shuffle(item.typewell_gr, item.well, 42)
    second, roll2 = EXP295.stable_circular_shuffle(item.typewell_gr, item.well, 42)
    assert roll1 == roll2 and roll1 != 0
    np.testing.assert_array_equal(first, second)


def test_stage_a_gate_is_conjunctive_and_fail_closed() -> None:
    config = load_config()
    stage = config["validation"]["stage_a_pass"]
    assert stage["minimum_target_in_grid_rate"] == 0.995
    assert stage["minimum_real_nll_gain_vs_circular_shuffle_nats_per_token"] == 0.05
    assert stage["minimum_real_rmse_gain_vs_exp209_ft"] == 0.25
    assert stage["maximum_peak_gpu_memory_gb"] == 14.0
    assert stage["maximum_fold_runtime_hours"] == 8.5
    failure_policy = " ".join(config["validation"]["failure_policy"])
    assert "without an in-experiment architecture" in failure_policy
    assert config["execution"]["stage_b_plan"]["prerequisite"] == "stage_a_all_pass"


def test_inference_is_fail_closed_until_stage_b_promotion() -> None:
    module = load_module(INFERENCE_SOURCE, "exp295_inference_contract")
    report = module.validate_disabled_inference(load_config())
    assert report["inference_enabled"] is False
    assert report["create_submission"] is False
    assert report["stage_c_prerequisite"] == "stage_b_lb5x_promotion_pass"
    unsafe = load_config()
    unsafe["inference"]["enabled"] = True
    with pytest.raises(ValueError, match="not approved"):
        module.validate_disabled_inference(unsafe)


def test_exact_torch_posterior_and_soft_label_gradient_when_torch_is_available() -> None:
    if not EXP295.TORCH_AVAILABLE:
        pytest.skip("local validation environment does not install PyTorch")
    torch = EXP295.torch
    config = load_config()
    spec = EXP295.StateSpec(
        grid=np.arange(1000.0, 1002.1, 0.35),
        rates=np.linspace(-0.05, 0.05, 5),
        suffix_index=np.arange(4, dtype=np.int64),
        dm=np.ones(4, dtype=np.float64),
        dz=np.full(4, -0.35, dtype=np.float64),
        start_p=1.0,
        init_rate=0.0,
        prefix_end=31,
        last_known_tvt=1000.35,
    )
    unary = torch.zeros((4, len(spec.grid)), dtype=torch.float32, requires_grad=True)
    truth_index = torch.tensor([1, 1, 6, 6], dtype=torch.long)
    target = torch.as_tensor(spec.grid[truth_index.numpy()], dtype=torch.float32)
    posterior, log_partition = EXP295.exact_forward_backward(unary, spec, config)
    assert torch.isfinite(log_partition)
    assert torch.allclose(posterior.sum(dim=1), torch.ones(4), atol=1e-5)
    label_emission = EXP295.gaussian_label_log_emission(target, spec, config)
    assert label_emission.shape == unary.shape
    assert torch.all(label_emission <= 0.0)
    loss = EXP295.SoftLabelStructuredNLL.apply(unary, target, spec, config)
    loss.backward()
    assert torch.isfinite(loss)
    assert loss >= 0.0
    assert torch.isfinite(unary.grad).all()
    assert torch.allclose(unary.grad.sum(dim=1), torch.zeros(4), atol=1e-5)
