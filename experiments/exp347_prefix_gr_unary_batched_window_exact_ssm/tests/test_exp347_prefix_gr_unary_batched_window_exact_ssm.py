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

ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = ROOT / "experiments" / "exp347_prefix_gr_unary_batched_window_exact_ssm"
TRAIN_SOURCE = EXP_DIR / (
    "exp347_prefix_gr_unary_batched_window_exact_ssm_compact_selfcontained_train.py"
)
INFERENCE_SOURCE = EXP_DIR / (
    "exp347_prefix_gr_unary_batched_window_exact_ssm_compact_selfcontained_inference.py"
)
os.environ["EXP347_IMPORT_ONLY"] = "1"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


EXP347 = load_module(TRAIN_SOURCE, "exp347_train_contract")


def load_config() -> dict:
    return yaml.safe_load((EXP_DIR / "config.yaml").read_text())


def synthetic_input(rows: int = 900, prefix_rows: int = 80):
    index = np.arange(rows, dtype=np.float64)
    tvt = 1000.0 + 0.15 * index
    tvt_input = tvt.copy()
    tvt_input[prefix_rows:] = np.nan
    type_tvt = np.arange(970.0, 1180.0, 0.35)
    type_gr = 70.0 + 9.0 * np.sin(type_tvt / 4.0)
    gr = np.interp(tvt, type_tvt, type_gr) + 1.5
    return EXP347.WellInput(
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


def write_synthetic_well(directory: Path, rows: int = 900, prefix_rows: int = 80) -> None:
    item, tvt = synthetic_input(rows, prefix_rows)
    horizontal = pd.DataFrame(
        {
            "MD": item.md,
            "X": item.x,
            "Y": item.y,
            "Z": item.z,
            "GR": item.gr,
            "TVT_input": item.tvt_input,
            "TVT": tvt,
            "candidate_leak": np.arange(rows),
        }
    )
    typewell = pd.DataFrame({"TVT": item.typewell_tvt, "GR": item.typewell_gr})
    horizontal.to_csv(directory / "well-a__horizontal_well.csv", index=False)
    typewell.to_csv(directory / "well-a__typewell.csv", index=False)


def test_scientific_gpu_and_implementation_contract_is_fixed() -> None:
    config = load_config()
    evidence = EXP347.validate_scientific_contract(config)
    cost = EXP347.validate_stage_a_cost_contract(config)
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
    execution = config["execution"]
    assert execution["implementation_approval_scope"] == "user_message_implement_exp347"
    assert execution["selected_stage"] == "implementation_only"
    assert execution["active_stage"] == "stage0_failed_branch_closed"
    assert execution["kaggle_push_approved"] is False
    assert (
        execution["kaggle_push_approval_scope"]
        == "user_message_execute_exp347_fixed16_stage0_only"
    )
    assert execution["stage0_run_completed"] is True
    assert execution["stage0_gate"]["passed"] is False
    assert execution["stage0_gate"]["report_sha256"] == (
        "e8a706ba9a75dff54b30b97f289255b002333cb76d2b2dfcac000cfdf56fe454"
    )
    assert execution["stage0_run"]["kernel_status"] == "COMPLETE"
    assert execution["stage0_run"]["posterior_max_abs_error"] > 1e-6
    assert execution["stage_a_gpu_approved"] is False
    assert execution["current_trained_fold_count"] == 0
    assert execution["canonical_train_notebook_adopted"] is True
    assert execution["inference_approved"] is False
    assert execution["submission_approved"] is False
    objective = config["model"]["training"]["objective"]
    assert objective["structured_label_nll_weight"] == 1.0
    assert objective["label_observation_sigma_ft"] == 0.35
    assert objective["local_true_state_ce_weight"] == 0.25
    assert objective["exact_dp_sweeps_per_window"] == 4
    batching = config["model"]["training"]["batching"]
    assert batching["windows_per_batch"] == 4
    assert batching["gradient_accumulation_steps"] == 1
    assert config["model"]["training"]["batch_size_well_windows"] == 4
    assert config["model"]["training"]["gradient_accumulation_windows"] == 1

    unsafe = copy.deepcopy(config)
    unsafe["model"]["training"]["windows"]["length_rows"] = 128
    with pytest.raises(ValueError, match="scientific contract changed"):
        EXP347.validate_scientific_contract(unsafe)


def test_groupkfold_identity_matches_exp202_sorted_well_contract() -> None:
    wells = [f"well-{index:03d}" for index in range(23)]
    actual = EXP347.build_fold_map(wells, 5).sort_values("well").reset_index(drop=True)
    from sklearn.model_selection import GroupKFold

    ordered = sorted(wells)
    expected_rows = []
    groups = np.asarray(ordered)
    dummy = np.zeros((len(ordered), 1), dtype=np.float32)
    for fold, (_, valid) in enumerate(GroupKFold(5).split(dummy, groups=groups)):
        expected_rows.extend({"well": ordered[index], "fold": fold} for index in valid)
    expected = pd.DataFrame(expected_rows).sort_values("well").reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, expected)


def test_window_slots_are_deterministic_nonoverlapping_and_inactive_when_unavailable() -> None:
    config = load_config()
    first = EXP347.select_window_slots("well-a", 80, 900, 0, config)
    second = EXP347.select_window_slots("well-a", 80, 900, 0, config)
    assert first == second
    assert len(first) == 3
    assert first[0]["start_row"] == 80
    active = [row for row in first if row["active"]]
    assert len(active) == 3
    intervals = sorted((row["start_row"], row["stop_row"]) for row in active)
    assert all(
        right_start >= left_stop
        for (_, left_stop), (right_start, _) in zip(
            intervals, intervals[1:], strict=False
        )
    )

    short = EXP347.select_window_slots("well-b", 80, 300, 0, config)
    assert short[0]["active"] is True and short[0]["scored_rows"] == 220
    assert [row["active"] for row in short[1:]] == [False, False]
    assert [row["scored_rows"] for row in short[1:]] == [0, 0]


def test_window_selection_source_has_no_target_dependent_access() -> None:
    source = inspect.getsource(EXP347.select_window_slots)
    forbidden = ("load_well_truth", "truth", "TVT", "formation", "oracle")
    assert not any(value in source for value in forbidden)
    assert "stable_uint64" in source


def test_schedule_freezes_before_teacher_boundary_and_encoder_stays_official(
    tmp_path: Path,
) -> None:
    write_synthetic_well(tmp_path)
    config = load_config()
    schedule = EXP347.build_window_schedule_manifest(
        ["well-a"], tmp_path, {"well-a": "fit"}, config
    )
    repeated = EXP347.build_window_schedule_manifest(
        ["well-a"], tmp_path, {"well-a": "fit"}, config
    )
    pd.testing.assert_frame_equal(schedule, repeated)
    assert len(schedule) == 8 * 3
    assert schedule.groupby(["well", "epoch"])["active"].sum().max() == 3

    boundary = EXP347.build_teacher_boundary_manifest(schedule, tmp_path, config)
    assert set(boundary["boundary_source"]) == {
        "official_prefix",
        "interior_teacher_loss_only",
    }
    assert boundary["encoder_tvt_input_source"].eq("official_prefix_only").all()
    interior = EXP347.window_keys_from_manifests(
        schedule, boundary, role="fit", epoch=0
    )[1]
    item, view, _, state = EXP347.prepare_training_window(interior, tmp_path, config)
    np.testing.assert_array_equal(view.tvt_input, item.tvt_input)
    assert np.isnan(view.tvt_input[80:]).all()
    assert state.suffix_index[0] == interior.start_row
    assert state.prefix_end == interior.start_row - 1
    assert state.last_known_tvt == pytest.approx(interior.boundary_tvt)
    assert state.init_rate == pytest.approx(interior.boundary_rate)


def test_mask_first_loader_never_selects_truth_or_candidate_columns(tmp_path: Path) -> None:
    write_synthetic_well(tmp_path, rows=100, prefix_rows=50)
    loaded = EXP347.load_well_input("well-a", tmp_path)
    assert not hasattr(loaded, "tvt")
    assert not hasattr(loaded, "candidate_leak")
    assert loaded.tvt_input.shape == (100,)
    truth = EXP347.load_well_truth("well-a", tmp_path)
    assert truth.tvt.shape == (100,)


def test_prepared_view_uses_fixed_exp209_grid_and_same_model_controls() -> None:
    item, _ = synthetic_input()
    config = load_config()
    real = EXP347.prepare_view(item, item.tvt_input, config, view_name="official")
    shuffled = EXP347.prepare_view(
        item,
        item.tvt_input,
        config,
        view_name="shuffle",
        typewell_control="shuffle",
    )
    assert real.horizontal_channels.shape == (3, len(item.md))
    assert real.state.rates.shape == (41,)
    assert np.diff(real.state.grid).mean() == pytest.approx(0.35)
    assert real.state.prefix_end == 79
    assert real.state.suffix_index[0] == 80
    assert not np.array_equal(real.typewell_channels, shuffled.typewell_channels)


def test_stage0_selection_is_four_windows_per_suffix_quartile() -> None:
    config = load_config()
    rows = [
        {
            "well": f"well-{index:03d}",
            "epoch": 0,
            "slot": 0,
            "active": True,
            "start_row": 80,
            "stop_row": 336,
            "scored_rows": 256,
            "suffix_rows": 300 + index * 10,
            "role": "fit",
        }
        for index in range(80)
    ]
    schedule = pd.DataFrame(rows)
    first = EXP347.select_fixed_benchmark_windows(schedule, config)
    second = EXP347.select_fixed_benchmark_windows(schedule, config)
    assert len(first) == 16 and first["well"].nunique() == 16
    assert first.groupby("suffix_length_quartile").size().to_dict() == {
        0: 4,
        1: 4,
        2: 4,
        3: 4,
    }
    pd.testing.assert_frame_equal(first, second)


def test_stage0_projection_uses_conservative_p10_rates() -> None:
    train = pd.DataFrame({"state_cells": [1000, 1000], "seconds": [1.0, 2.0]})
    window_forward = pd.DataFrame(
        {"state_cells": [1000, 1000], "seconds": [0.5, 1.0]}
    )
    unary = pd.DataFrame({"unary_positions": [100, 100], "seconds": [0.5, 1.0]})
    decode = pd.DataFrame({"state_cells": [1000, 1000], "seconds": [1.0, 2.0]})
    projected = EXP347.project_stage_a_runtime(
        train,
        window_forward,
        unary,
        decode,
        fit_window_state_cells_per_epoch=10_000,
        early_stop_window_state_cells_per_epoch=1000,
        valid_unary_positions=200,
        valid_decode_cells=2000,
        max_epochs=8,
    )
    assert projected["projected_fold_runtime_hours_conservative"] > (
        projected["projected_fold_runtime_hours_p50"]
    )


def test_stage0_and_stage_a_use_batched_dp_without_extra_control_models() -> None:
    stage0_source = inspect.getsource(EXP347.run_stage0_microbenchmark)
    training_source = inspect.getsource(EXP347.train_fold0_model)
    freeze_source = inspect.getsource(EXP347.freeze_outer_valid_predictions)
    parity_source = inspect.getsource(EXP347.run_scalar_batch_parity)
    assert "batched_window_training_loss(" in stage0_source
    assert "run_scalar_batch_parity(" in stage0_source
    assert "decode_unary_batch(" in stage0_source
    assert "fixed_window_batches(ordered)" in training_source
    assert "decode_unary_batch(" in freeze_source
    assert "copy.deepcopy" not in parity_source
    assert "nn.Parameter" in parity_source


def test_stage_selection_fail_closes_after_the_stage0_gate() -> None:
    config = load_config()
    assert EXP347.validate_selected_stage(config) == "implementation_only"
    benchmark = copy.deepcopy(config)
    benchmark["execution"]["selected_stage"] = "stage0_microbenchmark"
    with pytest.raises(ValueError, match="separate user approval"):
        EXP347.validate_selected_stage(benchmark)
    stage_a = copy.deepcopy(config)
    stage_a["execution"]["selected_stage"] = "stage_a_fold0"
    with pytest.raises(ValueError, match="separate user approval"):
        EXP347.validate_selected_stage(stage_a)


def test_inference_is_fail_closed_until_stage_b_promotion() -> None:
    module = load_module(INFERENCE_SOURCE, "exp347_inference_contract")
    report = module.validate_disabled_inference(load_config())
    assert report["inference_enabled"] is False
    assert report["create_submission"] is False
    assert report["stage_c_prerequisite"] == "stage_b_lb5x_promotion_pass"
    unsafe = load_config()
    unsafe["inference"]["enabled"] = True
    with pytest.raises(ValueError, match="not approved"):
        module.validate_disabled_inference(unsafe)


def test_window_objective_has_four_sweeps_and_teacher_boundary_is_loss_only() -> None:
    training_source = inspect.getsource(EXP347.batched_window_training_loss)
    terms_source = (
        inspect.getsource(EXP347.batched_soft_label_structured_terms)
        if EXP347.TORCH_AVAILABLE
        else TRAIN_SOURCE.read_text()
    )
    preparation_source = inspect.getsource(EXP347.prepare_training_window)
    assert "BatchedSoftLabelStructuredNLL.apply" in training_source
    assert "F.cross_entropy" in training_source
    assert terms_source.count("batched_exact_forward_backward(") >= 2
    assert "item.tvt_input" in preparation_source
    assert "key.boundary_tvt" not in preparation_source


def test_outer_valid_truth_is_joined_only_after_prediction_freeze() -> None:
    freeze_source = inspect.getsource(EXP347.freeze_outer_valid_predictions)
    readout_source = inspect.getsource(EXP347.post_freeze_readout)
    assert "load_well_truth" not in freeze_source
    assert '"truth_loaded_before_freeze": False' in freeze_source
    assert '"outer_valid_truth_access_count_before_freeze": 0' in freeze_source
    assert "load_well_truth" in readout_source
    assert "frozen.merge(truth_frame" in readout_source


def test_controls_and_sha_manifests_use_one_trained_model() -> None:
    config = load_config()
    assert config["execution"]["stage_a_plan"]["control_model_training"] == 0
    assert config["execution"]["control_or_parent_retraining"] is False
    source = inspect.getsource(EXP347.freeze_outer_valid_predictions)
    assert source.count("model_unary(model") == 2
    assert "torch.zeros_like(value)" in source
    assert source.count("decode_unary_batch(") == 3
    assert "real_unary_sha256" in source
    assert "real_posterior_sha256" in source
    assert "frozen_prediction_decompressed_sha256" in source


def test_exact_torch_posterior_and_structured_gradient_when_available() -> None:
    if not EXP347.TORCH_AVAILABLE:
        pytest.skip("local validation environment does not install PyTorch")
    torch = EXP347.torch
    config = load_config()
    spec = EXP347.StateSpec(
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
    truth_index = torch.tensor([1, 1, 4, 4], dtype=torch.long)
    target = torch.as_tensor(spec.grid[truth_index.numpy()], dtype=torch.float32)
    posterior, log_partition = EXP347.exact_forward_backward(unary, spec, config)
    assert torch.isfinite(log_partition)
    assert torch.allclose(posterior.sum(dim=1), torch.ones(4), atol=1e-5)
    loss = EXP347.SoftLabelStructuredNLL.apply(unary, target, spec, config)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(unary.grad).all()
    assert torch.allclose(unary.grad.sum(dim=1), torch.zeros(4), atol=1e-5)


def test_batched_exact_dp_matches_scalar_with_position_row_and_dummy_padding() -> None:
    if not EXP347.TORCH_AVAILABLE:
        pytest.skip("local validation environment does not install PyTorch")
    torch = EXP347.torch
    config = load_config()

    def state(rows: int, positions: int, offset: float) -> object:
        grid = offset + np.arange(positions, dtype=np.float64) * 0.35
        return EXP347.StateSpec(
            grid=grid,
            rates=np.linspace(-0.05, 0.05, 5),
            suffix_index=np.arange(rows, dtype=np.int64),
            dm=np.linspace(1.0, 1.3, rows),
            dz=np.full(rows, -0.35, dtype=np.float64),
            start_p=1.0,
            init_rate=0.0,
            prefix_end=31,
            last_known_tvt=float(grid[1]),
        )

    states = [state(4, 7, 1000.0), state(3, 6, 1100.0), state(2, 5, 1200.0)]
    unaries = [
        torch.linspace(-0.2, 0.3, len(spec.suffix_index) * len(spec.grid)).reshape(
            len(spec.suffix_index), len(spec.grid)
        )
        for spec in states
    ]
    batch = EXP347.build_batched_state_spec(states, required_batch_size=4)
    padded = EXP347.pad_unary_batch(unaries, batch).detach().requires_grad_(True)
    posterior, partition = EXP347.batched_exact_forward_backward(padded, batch, config)
    for index, (unary, spec) in enumerate(zip(unaries, states, strict=True)):
        scalar_posterior, scalar_partition = EXP347.exact_forward_backward(
            unary, spec, config
        )
        actual = posterior[index, : len(spec.suffix_index), : len(spec.grid)]
        assert torch.allclose(actual, scalar_posterior, atol=1e-6, rtol=0.0)
        assert torch.allclose(partition[index], scalar_partition, atol=1e-6, rtol=0.0)
    row_mask = torch.as_tensor(batch.row_mask, dtype=torch.bool)
    position_mask = torch.as_tensor(batch.position_mask, dtype=torch.bool)
    invalid = ~(row_mask[:, :, None] & position_mask[:, None, :])
    assert torch.count_nonzero(posterior.masked_select(invalid)) == 0
    assert batch.active_mask.tolist() == [True, True, True, False]


def test_fixed_batch_formation_preserves_consecutive_frozen_order() -> None:
    keys = [
        EXP347.WindowKey(
            well=f"well-{index}",
            epoch=0,
            slot=index % 3,
            start_row=80 + index,
            stop_row=336 + index,
            scored_rows=256,
            boundary_source="official_prefix",
            boundary_tvt=1000.0,
            boundary_rate=0.0,
            boundary_rate_index=20,
        )
        for index in range(10)
    ]
    batches = EXP347.fixed_window_batches(keys)
    assert [[key.well for key in batch] for batch in batches] == [
        ["well-0", "well-1", "well-2", "well-3"],
        ["well-4", "well-5", "well-6", "well-7"],
        ["well-8", "well-9"],
    ]
    with pytest.raises(ValueError, match="four consecutive"):
        EXP347.fixed_window_batches(keys, batch_size=2)
