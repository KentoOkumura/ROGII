from __future__ import annotations

import copy
import importlib.util
import inspect
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "experiments" / "exp331_prefix_gr_unary_local_ce_exact_ssm"
TRAIN_SOURCE = EXP_DIR / (
    "exp331_prefix_gr_unary_local_ce_exact_ssm_compact_selfcontained_train.py"
)
INFERENCE_SOURCE = EXP_DIR / (
    "exp331_prefix_gr_unary_local_ce_exact_ssm_compact_selfcontained_inference.py"
)
os.environ["EXP331_IMPORT_ONLY"] = "1"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


EXP331 = load_module(TRAIN_SOURCE, "exp331_train_contract")


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
    return EXP331.WellInput(
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
    evidence = EXP331.validate_scientific_contract(config)
    cost = EXP331.validate_stage_a_cost_contract(config)
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
    assert config["execution"]["implementation_approval_scope"] == (
        "user_message_implement_exp331"
    )
    assert config["execution"]["selected_stage"] == "implementation_only"
    assert config["execution"]["kaggle_push_approved"] is False
    assert config["execution"]["kaggle_push_approval_scope"] == (
        "user_message_start_exp331_stage_a_fold0_only"
    )
    assert config["execution"]["stage_a_gpu_approved"] is False
    assert config["execution"]["stage_a_gpu_approval_scope"] == (
        "user_message_start_exp331_stage_a_fold0_only"
    )
    assert config["execution"]["stage0_gate"] == {
        "passed": True,
        "report_sha256": (
            "401d98f2cdc9ced437d66fc02bbe49b9287d4772e4d9036719c573a90b785c59"
        ),
        "note": (
            "Kaggle T4 version 1 passed: conservative fold projection 4.516839 h "
            "<= 8.5 h and peak GPU memory 1.924052 GB <= 14 GB. Stage A still "
            "requires separate approval."
        ),
    }
    assert config["execution"]["canonical_train_notebook_adopted"] is True
    assert config["execution"]["canonical_inference_notebook_adopted"] is False
    assert config["execution"]["inference_approved"] is False
    assert config["execution"]["submission_approved"] is False
    assert config["execution"]["current_trained_fold_count"] == 1
    assert config["execution"]["stage_a_gate"]["passed"] is False
    assert config["execution"]["stage_a_gate"]["failed_checks"] == [
        "real_rmse_vs_exp209",
        "well_p95_non_regression",
        "worst_well_regression",
    ]
    assert config["execution"]["stage_a_gate"]["stage_b_blocked"] is True
    assert config["model"]["training"]["objective"] == {
        "name": "hard_nearest_state_local_cross_entropy",
        "local_true_state_ce_weight": 1.0,
        "structured_label_nll_weight": 0.0,
        "state_space_calls_in_optimizer_graph": 0,
        "state_space_calls_for_early_stopping": 0,
    }

    unsafe = copy.deepcopy(config)
    unsafe["model"]["state_space"]["band_pad"] = 120.0
    with pytest.raises(ValueError, match="scientific contract changed"):
        EXP331.validate_scientific_contract(unsafe)


def test_groupkfold_identity_matches_exp202_sorted_well_contract() -> None:
    wells = [f"well-{index:03d}" for index in range(23)]
    actual = EXP331.build_fold_map(wells, 5).sort_values("well").reset_index(drop=True)
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
    official_end = EXP331.prefix_end_index(item.tvt_input)
    assert official_end == 79
    view, cut_end = EXP331.make_view_tvt_input(item.tvt_input, 32)
    assert cut_end == 47
    assert np.isfinite(view[:48]).all()
    assert np.isnan(view[48:]).all()
    with pytest.raises(ValueError, match="fewer than 32"):
        EXP331.make_view_tvt_input(item.tvt_input, 64)


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
    loaded = EXP331.load_well_input(well, tmp_path)
    assert not hasattr(loaded, "tvt")
    assert not hasattr(loaded, "candidate_leak")
    assert loaded.tvt_input.shape == (40,)
    truth = EXP331.load_well_truth(well, tmp_path)
    np.testing.assert_allclose(truth.tvt, horizontal["TVT"])


def test_robust_preprocessing_and_typewell_interpolation_forbid_extrapolation() -> None:
    values = np.array([1.0, 2.0, np.nan, 4.0])
    normalized, missing, center, scale = EXP331.robust_normalize(values)
    assert normalized[2] == 0.0
    assert missing.tolist() == [0.0, 0.0, 1.0, 0.0]
    assert center == 2.0
    assert scale >= 1.0

    interpolated, inside = EXP331.interpolate_no_extrapolation(
        np.array([-1.0, 0.5, 2.0]), np.array([0.0, 1.0]), np.array([10.0, 20.0])
    )
    assert inside.tolist() == [False, True, False]
    assert interpolated.tolist() == [0.0, 15.0, 0.0]


def test_huber_prefix_summary_has_neutral_fallback_and_recovers_affine_relation() -> None:
    x = np.linspace(20.0, 120.0, 64)
    y = 1.2 * x + 7.0
    summary, count = EXP331.huber_affine_summary(x, y)
    assert count == 64
    assert summary[-1] == 1.0
    assert summary[0] == pytest.approx(1.2, abs=1e-5)

    neutral, count = EXP331.huber_affine_summary(x[:20], y[:20])
    assert count == 20
    np.testing.assert_array_equal(neutral, np.zeros(5, dtype=np.float32))


def test_prepared_view_uses_only_one_well_and_fixed_exp209_grid() -> None:
    item, _ = synthetic_input()
    config = load_config()
    real = EXP331.prepare_view(item, item.tvt_input, config, view_name="official")
    shuffled = EXP331.prepare_view(
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
    first, roll1 = EXP331.stable_circular_shuffle(item.typewell_gr, item.well, 42)
    second, roll2 = EXP331.stable_circular_shuffle(item.typewell_gr, item.well, 42)
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


def test_stage0_selection_is_fixed_four_views_per_suffix_quartile() -> None:
    config = load_config()
    rows = []
    for index in range(80):
        rows.append(
            {
                "well": f"well-{index:03d}",
                "view_name": "official",
                "offset_rows": 0,
                "hidden_rows": 100 + index * 10,
                "eligible": True,
            }
        )
    manifest = pd.DataFrame(rows)
    first = EXP331.select_fixed_benchmark_views(manifest, manifest["well"], config)
    second = EXP331.select_fixed_benchmark_views(manifest, manifest["well"], config)
    assert len(first) == 16
    assert first.groupby("suffix_length_quartile").size().to_dict() == {
        0: 4,
        1: 4,
        2: 4,
        3: 4,
    }
    pd.testing.assert_frame_equal(first, second)


def test_stage0_projection_uses_conservative_p10_rates() -> None:
    train = pd.DataFrame({"unary_positions": [100, 100], "seconds": [1.0, 2.0]})
    forward = pd.DataFrame({"unary_positions": [100, 100], "seconds": [0.5, 1.0]})
    decode = pd.DataFrame({"state_cells": [1000, 1000], "seconds": [1.0, 2.0]})
    projected = EXP331.project_stage_a_runtime(
        train,
        forward,
        decode,
        fit_unary_positions_per_epoch=1000,
        early_stop_unary_positions_per_epoch=100,
        valid_unary_positions=200,
        valid_decode_cells=2000,
        max_epochs=8,
    )
    assert projected["projected_fold_runtime_hours_conservative"] > (
        projected["projected_fold_runtime_hours_p50"]
    )


def test_stage_selection_blocks_gpu_work_without_separate_approval() -> None:
    config = load_config()
    assert EXP331.validate_selected_stage(config) == "implementation_only"
    benchmark = copy.deepcopy(config)
    benchmark["execution"]["selected_stage"] = "stage0_microbenchmark"
    benchmark["execution"]["kaggle_push_approved"] = False
    with pytest.raises(ValueError, match="separate user approval"):
        EXP331.validate_selected_stage(benchmark)
    stage_a = copy.deepcopy(config)
    stage_a["execution"]["selected_stage"] = "stage_a_fold0"
    stage_a["execution"]["stage_a_gpu_approved"] = False
    with pytest.raises(ValueError, match="separate user approval"):
        EXP331.validate_selected_stage(stage_a)


def test_inference_is_fail_closed_until_stage_b_promotion() -> None:
    module = load_module(INFERENCE_SOURCE, "exp331_inference_contract")
    report = module.validate_disabled_inference(load_config())
    assert report["inference_enabled"] is False
    assert report["create_submission"] is False
    assert report["stage_c_prerequisite"] == "stage_b_lb5x_promotion_pass"
    unsafe = load_config()
    unsafe["inference"]["enabled"] = True
    with pytest.raises(ValueError, match="not approved"):
        module.validate_disabled_inference(unsafe)


def test_training_and_early_stop_source_have_no_state_space_call() -> None:
    training_source = inspect.getsource(EXP331.training_loss)
    early_source = inspect.getsource(EXP331.evaluate_early_stop_loss)
    forbidden = ("exact_forward_backward", "decode_unary", "SoftLabelStructuredNLL")
    assert not any(name in training_source for name in forbidden)
    assert not any(name in early_source for name in forbidden)
    assert "F.cross_entropy" in training_source
    assert "F.cross_entropy" in early_source


def test_outer_valid_truth_is_joined_only_after_prediction_freeze() -> None:
    freeze_source = inspect.getsource(EXP331.freeze_outer_valid_predictions)
    readout_source = inspect.getsource(EXP331.post_freeze_readout)
    assert "load_well_truth" not in freeze_source
    assert '"truth_loaded_before_freeze": False' in freeze_source
    assert '"outer_valid_truth_access_count_before_freeze": 0' in freeze_source
    assert "load_well_truth" in readout_source
    assert "frozen.merge(truth_frame" in readout_source


def test_controls_and_sha_manifests_use_one_trained_model() -> None:
    config = load_config()
    assert config["execution"]["stage_a_plan"]["control_model_training"] == 0
    assert config["execution"]["control_or_parent_retraining"] is False
    source = inspect.getsource(EXP331.freeze_outer_valid_predictions)
    assert source.count("model_unary(model") == 2
    assert "torch.zeros_like(real_unary)" in source
    assert "real_unary_sha256" in source
    assert "shuffle_unary_sha256" in source
    assert "real_posterior_sha256" in source
    assert "frozen_prediction_decompressed_sha256" in source


def test_long_tail_and_hidden_like_subgroup_metrics_are_fixed() -> None:
    readout = pd.DataFrame(
        {
            "well": ["a", "a", "b", "b"],
            "md_since": [500.0, 1200.0, 1500.0, 200.0],
            "truth_tvt": [0.0, 1.0, 2.0, 3.0],
            "real_prediction": [0.0, 1.0, 2.0, 3.0],
            "shuffle_prediction": [1.0, 2.0, 3.0, 4.0],
            "geometry_prediction": [2.0, 3.0, 4.0, 5.0],
            "exp209_prediction": [3.0, 4.0, 5.0, 6.0],
        }
    )
    metrics = EXP331.subgroup_metric_table(
        readout,
        {"hidden_like_spatial": {"a"}, "hidden_like_typewell_purged": {"b"}},
    )
    assert set(metrics["subgroup"]) == {
        "distance_1000_plus",
        "hidden_like_spatial",
        "hidden_like_typewell_purged",
    }
    assert set(metrics["candidate"]) == {
        "real_gr",
        "circular_shuffle",
        "geometry_only",
        "exp209",
    }
    assert metrics.loc[metrics["candidate"] == "real_gr", "rmse"].eq(0.0).all()


def test_exact_torch_posterior_is_evaluation_only_when_torch_is_available() -> None:
    if not EXP331.TORCH_AVAILABLE:
        pytest.skip("local validation environment does not install PyTorch")
    torch = EXP331.torch
    config = load_config()
    spec = EXP331.StateSpec(
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
    posterior, log_partition = EXP331.exact_forward_backward(unary, spec, config)
    assert torch.isfinite(log_partition)
    assert torch.allclose(posterior.sum(dim=1), torch.ones(4), atol=1e-5)
    assert not hasattr(EXP331, "SoftLabelStructuredNLL")
