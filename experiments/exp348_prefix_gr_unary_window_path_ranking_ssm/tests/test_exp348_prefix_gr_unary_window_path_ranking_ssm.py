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
EXP_DIR = ROOT / "experiments" / "exp348_prefix_gr_unary_window_path_ranking_ssm"
TRAIN_SOURCE = EXP_DIR / (
    "exp348_prefix_gr_unary_window_path_ranking_ssm_compact_selfcontained_train.py"
)
INFERENCE_SOURCE = EXP_DIR / (
    "exp348_prefix_gr_unary_window_path_ranking_ssm_compact_selfcontained_inference.py"
)
os.environ["EXP348_IMPORT_ONLY"] = "1"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


EXP348 = load_module(TRAIN_SOURCE, "exp348_train_contract")


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
    return EXP348.WellInput(
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
    evidence = EXP348.validate_scientific_contract(config)
    cost = EXP348.validate_stage_a_cost_contract(config)
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
    assert execution["implementation_approval_scope"] == "user_message_implement_exp348"
    assert execution["selected_stage"] == "stage0_microbenchmark"
    assert execution["active_stage"] == "stage0_failed_branch_closed"
    assert execution["kaggle_push_approved"] is True
    assert (
        execution["kaggle_push_approval_scope"]
        == "user_message_execute_exp348_stage0"
    )
    assert execution["kaggle_push_retry_approved"] is True
    assert (
        execution["kaggle_push_retry_approval_scope"]
        == "user_message_quota_recovered_execute_exp348"
    )
    assert execution["stage0_run_completed"] is True
    assert execution["stage0_gate"]["passed"] is False
    assert execution["stage0_gate"]["report_sha256"] == (
        "2ba5d21934ca1ce49b2e384dd1ea7414f618e4926b4167ac0991d2787fe34c9b"
    )
    assert execution["stage0_gate"]["technical_pass"] is True
    assert execution["stage0_gate"]["learning_pass"] is False
    assert execution["stage0_gate"]["runtime_pass"] is False
    assert execution["stage0_gate"]["memory_pass"] is True
    assert execution["last_push_attempt"]["outcome"] == (
        "rejected_before_kernel_version_creation"
    )
    assert execution["last_push_attempt"]["kernel_version_created"] is False
    assert execution["stage0_kaggle_run"]["id_no"] == 128524049
    assert execution["stage0_kaggle_run"]["version"] == 2
    assert execution["stage0_kaggle_run"]["accelerator"] == "NvidiaTeslaT4"
    assert execution["stage0_kaggle_run"]["status"] == "complete"
    assert execution["stage0_kaggle_run"]["monitoring_paused_by_user"] is False
    assert execution["stage0_kaggle_history"][0]["version"] == 1
    assert execution["stage0_kaggle_history"][0]["training_started"] is False
    assert execution["stage0_version_2_retry"]["scientific_contract_changed"] is False
    assert execution["stage_a_gpu_approved"] is False
    assert execution["current_trained_fold_count"] == 0
    assert execution["canonical_train_notebook_adopted"] is True
    assert execution["inference_approved"] is False
    assert execution["submission_approved"] is False
    objective = config["model"]["training"]["objective"]
    assert objective["path_ranking_weight"] == 1.0
    assert objective["label_observation_sigma_ft"] == 0.35
    assert objective["local_true_state_ce_weight"] == 0.25
    assert objective["margin_per_valid_row"] == 0.05
    assert objective["exact_partition_sweeps_per_window"] == 0
    negatives = config["model"]["training"]["negative_paths"]
    assert negatives["maximum_count"] == 16
    assert negatives["minimum_unique_count"] == 12
    assert negatives["model_dependent_hard_negative_mining"] is False

    unsafe = copy.deepcopy(config)
    unsafe["model"]["training"]["windows"]["length_rows"] = 128
    with pytest.raises(ValueError, match="scientific contract changed"):
        EXP348.validate_scientific_contract(unsafe)


def test_groupkfold_identity_matches_exp202_sorted_well_contract() -> None:
    wells = [f"well-{index:03d}" for index in range(23)]
    actual = EXP348.build_fold_map(wells, 5).sort_values("well").reset_index(drop=True)
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
    first = EXP348.select_window_slots("well-a", 80, 900, 0, config)
    second = EXP348.select_window_slots("well-a", 80, 900, 0, config)
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

    short = EXP348.select_window_slots("well-b", 80, 300, 0, config)
    assert short[0]["active"] is True and short[0]["scored_rows"] == 220
    assert [row["active"] for row in short[1:]] == [False, False]
    assert [row["scored_rows"] for row in short[1:]] == [0, 0]


def test_window_selection_source_has_no_target_dependent_access() -> None:
    source = inspect.getsource(EXP348.select_window_slots)
    forbidden = ("load_well_truth", "truth", "TVT", "formation", "oracle")
    assert not any(value in source for value in forbidden)
    assert "stable_uint64" in source


def test_schedule_freezes_before_teacher_boundary_and_encoder_stays_official(
    tmp_path: Path,
) -> None:
    write_synthetic_well(tmp_path)
    config = load_config()
    schedule = EXP348.build_window_schedule_manifest(
        ["well-a"], tmp_path, {"well-a": "fit"}, config
    )
    repeated = EXP348.build_window_schedule_manifest(
        ["well-a"], tmp_path, {"well-a": "fit"}, config
    )
    pd.testing.assert_frame_equal(schedule, repeated)
    assert len(schedule) == 8 * 3
    assert schedule.groupby(["well", "epoch"])["active"].sum().max() == 3

    boundary = EXP348.build_teacher_boundary_manifest(schedule, tmp_path, config)
    assert set(boundary["boundary_source"]) == {
        "official_prefix",
        "interior_teacher_loss_only",
    }
    assert boundary["encoder_tvt_input_source"].eq("official_prefix_only").all()
    interior = EXP348.window_keys_from_manifests(
        schedule, boundary, role="fit", epoch=0
    )[1]
    item, view, _, state = EXP348.prepare_training_window(interior, tmp_path, config)
    np.testing.assert_array_equal(view.tvt_input, item.tvt_input)
    assert np.isnan(view.tvt_input[80:]).all()
    assert state.suffix_index[0] == interior.start_row
    assert state.prefix_end == interior.start_row - 1
    assert state.last_known_tvt == pytest.approx(interior.boundary_tvt)
    assert state.init_rate == pytest.approx(interior.boundary_rate)


def test_mask_first_loader_never_selects_truth_or_candidate_columns(tmp_path: Path) -> None:
    write_synthetic_well(tmp_path, rows=100, prefix_rows=50)
    loaded = EXP348.load_well_input("well-a", tmp_path)
    assert not hasattr(loaded, "tvt")
    assert not hasattr(loaded, "candidate_leak")
    assert loaded.tvt_input.shape == (100,)
    truth = EXP348.load_well_truth("well-a", tmp_path)
    assert truth.tvt.shape == (100,)


def test_exp209_window_alignment_uses_generated_well_row_ids_without_raw_id_column(
    tmp_path: Path,
) -> None:
    write_synthetic_well(tmp_path, rows=100, prefix_rows=50)
    item = EXP348.load_well_input("well-a", tmp_path)
    assert "id" not in pd.read_csv(item.horizontal_path, nrows=0).columns
    suffix_index = np.asarray([50, 52, 51], dtype=np.int64)
    baseline = pd.DataFrame(
        {
            "id": ["well-a_51", "well-a_50", "well-a_52"],
            "well": ["well-a", "well-a", "well-a"],
            "exp209_prediction": [1051.0, 1050.0, 1052.0],
        }
    )
    aligned = EXP348.align_exp209_window_template(item, baseline, suffix_index)
    np.testing.assert_array_equal(aligned, np.asarray([1050.0, 1052.0, 1051.0]))


def test_prepared_view_uses_fixed_exp209_grid_and_same_model_controls() -> None:
    item, _ = synthetic_input()
    config = load_config()
    real = EXP348.prepare_view(item, item.tvt_input, config, view_name="official")
    shuffled = EXP348.prepare_view(
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
    first = EXP348.select_fixed_benchmark_windows(schedule, config)
    second = EXP348.select_fixed_benchmark_windows(schedule, config)
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
    projected = EXP348.project_stage_a_runtime(
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


def test_stage_selection_allows_only_the_approved_stage0_scope() -> None:
    config = load_config()
    assert EXP348.validate_selected_stage(config) == "stage0_microbenchmark"
    benchmark = copy.deepcopy(config)
    benchmark["execution"]["selected_stage"] = "stage0_microbenchmark"
    benchmark["execution"]["kaggle_push_approved"] = False
    with pytest.raises(ValueError, match="separate user approval"):
        EXP348.validate_selected_stage(benchmark)
    stage_a = copy.deepcopy(config)
    stage_a["execution"]["selected_stage"] = "stage_a_fold0"
    with pytest.raises(ValueError, match="separate user approval"):
        EXP348.validate_selected_stage(stage_a)


def test_inference_is_fail_closed_until_stage_b_promotion() -> None:
    module = load_module(INFERENCE_SOURCE, "exp348_inference_contract")
    report = module.validate_disabled_inference(load_config())
    assert report["inference_enabled"] is False
    assert report["create_submission"] is False
    assert report["stage_c_prerequisite"] == "stage_b_lb5x_promotion_pass"
    unsafe = load_config()
    unsafe["inference"]["enabled"] = True
    with pytest.raises(ValueError, match="not approved"):
        module.validate_disabled_inference(unsafe)


def test_window_objective_is_gather_only_ranking_and_teacher_boundary_is_loss_only() -> None:
    training_source = inspect.getsource(EXP348.window_path_ranking_loss)
    preparation_source = inspect.getsource(EXP348.prepare_training_window)
    assert "score_legal_path" in training_source
    assert "F.softplus" in training_source
    assert "F.cross_entropy" in training_source
    assert "exact_forward_backward" not in training_source
    assert "SoftLabelStructuredNLL" not in training_source
    assert "item.tvt_input" in preparation_source
    assert "key.boundary_tvt" not in preparation_source


def test_path_bank_is_frozen_before_model_construction_and_stage0_has_holdout() -> None:
    stage0_source = inspect.getsource(EXP348.run_stage0_microbenchmark)
    stage_a_source = inspect.getsource(EXP348.run_stage_a)
    training_source = inspect.getsource(EXP348.train_fold0_model)
    assert stage0_source.index("build_frozen_path_banks(") < stage0_source.index(
        "PrefixConditionedUnary(config)"
    )
    assert stage_a_source.index("build_frozen_path_banks(") < stage_a_source.index(
        "train_fold0_model("
    )
    assert '"early_holdout"' in stage0_source
    assert "positive_top1_rate" in stage0_source
    assert "positive_max_negative_margin" in stage0_source
    assert "path banks must be fully frozen before model construction" in training_source


def test_path_bank_contract_has_fixed_families_dedup_and_fail_closed_minimum() -> None:
    source = inspect.getsource(EXP348.build_window_path_bank)
    assert "position_grid_index_offsets" in source
    assert "constant_rate_index_offsets" in source
    assert "midpoint_rate_index_pulses" in source
    assert "saved_exp209_official_prefix_path" in source
    assert "geometry_only_fixed_grammar_path" in source
    assert "duplicate_positive_or_earlier_negative" in source
    assert "unique negative paths" in source
    assert "model_unary" not in source
    assert "outer_valid" not in source


def test_outer_valid_truth_is_joined_only_after_prediction_freeze() -> None:
    freeze_source = inspect.getsource(EXP348.freeze_outer_valid_predictions)
    readout_source = inspect.getsource(EXP348.post_freeze_readout)
    assert "load_well_truth" not in freeze_source
    assert '"truth_loaded_before_freeze": False' in freeze_source
    assert '"outer_valid_truth_access_count_before_freeze": 0' in freeze_source
    assert "load_well_truth" in readout_source
    assert "frozen.merge(truth_frame" in readout_source


def test_controls_and_sha_manifests_use_one_trained_model() -> None:
    config = load_config()
    assert config["execution"]["stage_a_plan"]["control_model_training"] == 0
    assert config["execution"]["control_or_parent_retraining"] is False
    source = inspect.getsource(EXP348.freeze_outer_valid_predictions)
    assert source.count("model_unary(model") == 2
    assert "torch.zeros_like(real_unary)" in source
    assert "real_unary_sha256" in source
    assert "real_posterior_sha256" in source
    assert "frozen_prediction_decompressed_sha256" in source


def test_exact_decoder_and_gather_ranking_gradient_when_available() -> None:
    if not EXP348.TORCH_AVAILABLE:
        pytest.skip("local validation environment does not install PyTorch")
    torch = EXP348.torch
    config = load_config()
    spec = EXP348.StateSpec(
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
    posterior, log_partition = EXP348.exact_forward_backward(unary, spec, config)
    assert torch.isfinite(log_partition)
    assert torch.allclose(posterior.sum(dim=1), torch.ones(4), atol=1e-5)
    positive = EXP348.LegalPath(
        family="positive",
        candidate_order=-1,
        boundary_position_index=0,
        boundary_rate_index=2,
        position_index=np.asarray([1, 2, 3, 4], dtype=np.int16),
        rate_index=np.asarray([2, 2, 2, 2], dtype=np.int16),
        fixed_log_potential_sum=0.0,
        template_sha256="a" * 64,
        content_sha256="b" * 64,
    )
    negative = EXP348.LegalPath(
        family="negative",
        candidate_order=0,
        boundary_position_index=1,
        boundary_rate_index=2,
        position_index=np.asarray([2, 3, 4, 5], dtype=np.int16),
        rate_index=np.asarray([2, 2, 2, 2], dtype=np.int16),
        fixed_log_potential_sum=0.0,
        template_sha256="c" * 64,
        content_sha256="d" * 64,
    )
    assert EXP348.path_is_legal(positive, spec, config)
    assert EXP348.path_is_legal(negative, spec, config)
    positive_score = EXP348.score_legal_path(unary, positive, spec, config)
    negative_score = EXP348.score_legal_path(unary, negative, spec, config)
    loss = torch.nn.functional.softplus(0.05 + negative_score - positive_score)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(unary.grad).all()
    assert torch.count_nonzero(unary.grad).item() > 0
