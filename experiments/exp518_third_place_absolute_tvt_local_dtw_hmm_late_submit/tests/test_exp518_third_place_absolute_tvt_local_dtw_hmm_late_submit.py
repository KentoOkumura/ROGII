from __future__ import annotations

import importlib.util
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = ROOT / "experiments" / "exp518_third_place_absolute_tvt_local_dtw_hmm_late_submit"
TRAIN_SOURCE = (
    EXP_DIR
    / "exp518_third_place_absolute_tvt_local_dtw_hmm_late_submit_compact_selfcontained_train.py"
)
INFERENCE_SOURCE = (
    EXP_DIR
    / "exp518_third_place_absolute_tvt_local_dtw_hmm_late_submit_compact_selfcontained_inference.py"
)


def load_source(path: Path, name: str):
    os.environ["EXP518_IMPORT_ONLY"] = "1"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TRAIN = load_source(TRAIN_SOURCE, "exp518_train_test_module")
INFERENCE = load_source(INFERENCE_SOURCE, "exp518_inference_test_module")


def synthetic_horizontal(include_truth: bool = False) -> pd.DataFrame:
    rows = 12
    frame = pd.DataFrame(
        {
            "MD": np.arange(rows, dtype=float),
            "Z": -np.arange(rows, dtype=float),
            "GR": 100.0 + np.sin(np.arange(rows) / 2.0),
            "TVT_input": [1000.0 + index if index < 6 else np.nan for index in range(rows)],
        }
    )
    if include_truth:
        frame["TVT"] = 1000.0 + np.arange(rows)
    return frame


def synthetic_typewell() -> pd.DataFrame:
    tvt = np.arange(980.0, 1040.5, 0.5)
    return pd.DataFrame({"TVT": tvt, "GR": 100.0 + np.sin((tvt - 1000.0) / 2.0)})


def synthetic_guide() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "well_id": "synthetic",
            "row_idx": np.arange(6, 12, dtype=np.int64),
            "student_t_df4_on_exp209_absolute_tvt_hmm_tvt": np.arange(1006.0, 1012.0),
            "student_t_df4_on_exp209_absolute_tvt_hmm_std": np.full(6, 1.0),
        }
    )


def test_frozen_contract_uses_absolute_tvt_and_blocks_inference() -> None:
    config = TRAIN.load_experiment_config()
    TRAIN.validate_frozen_contract(config)
    state = config["model"]["state"]
    assert state["coordinate"] == "absolute_tvt_around_absolute_hmm_guide"
    assert state["guide_band_pad_ft"] == 60.0
    assert state["position_step_ft"] == 1.0
    assert "offset_min_ft" not in state
    assert config["method_fidelity"]["classification"] == "proxy"
    assert config["late_submission"]["inference_requires_oof_reproduction_and_user_confirmation"]
    assert "LATE SUBMIT" in config["late_submission"]["message"]
    assert math.isclose(sum(config["model"]["family_weights"].values()), 1.0)


def test_nearest_transition_probabilities_are_normalized() -> None:
    logs = TRAIN._nearest_transition_logs(
        np.asarray([0.0, 2.0]),
        np.asarray([-0.02, 0.0, 0.02]),
        sigma_per_sqrt_ft=0.002,
        momentum=0.999,
    )
    for time_index in range(logs.shape[0]):
        for source in range(logs.shape[1]):
            finite = logs[time_index, source] > -1e17
            assert np.isclose(np.exp(logs[time_index, source, finite]).sum(), 1.0)


def dense_transition(
    rate_log: np.ndarray,
    bias_log: np.ndarray,
    position_b0: np.ndarray,
    position_log: np.ndarray,
    reference_log: np.ndarray,
    time_index: int,
    p_count: int,
) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    r_count = rate_log.shape[1]
    b_count = bias_log.shape[1]
    f_count = reference_log.shape[0]
    states = [
        (p, r, b, f)
        for p in range(p_count)
        for r in range(r_count)
        for b in range(b_count)
        for f in range(f_count)
    ]
    transition = np.zeros((len(states), len(states)), dtype=np.float64)
    for source_index, (p1, r1, b1, f1) in enumerate(states):
        for target_index, (p2, r2, b2, f2) in enumerate(states):
            if abs(r2 - r1) > 1 or abs(b2 - b1) > 1:
                continue
            rate_value = rate_log[time_index, r1, r2 - r1 + 1]
            bias_value = bias_log[time_index, b1, b2 - b1 + 1]
            if rate_value < -1e17 or bias_value < -1e17 or reference_log[f1, f2] < -1e17:
                continue
            center = int(position_b0[time_index, r2])
            for kernel_index in range(5):
                if p2 == p1 + center - 2 + kernel_index:
                    transition[source_index, target_index] = math.exp(
                        rate_value
                        + bias_value
                        + reference_log[f1, f2]
                        + position_log[time_index, r2, kernel_index]
                    )
                    break
    return transition, states


def test_absolute_tvt_forward_backward_matches_dense_sum_product() -> None:
    config = TRAIN.load_experiment_config()
    dm = np.asarray([0.0, 1.0], dtype=np.float64)
    positions = np.asarray([999.0, 1000.0, 1001.0], dtype=np.float64)
    rates = np.asarray([-0.01, 0.01], dtype=np.float64)
    biases = np.asarray([-1.0, 1.0], dtype=np.float64)
    rate_log, bias_log, position_b0, position_log, reference_log = (
        TRAIN.prepare_transition_logs(
            dm,
            rates,
            biases,
            config,
            reference_count=2,
            family="exp417_base_reconstruction",
        )
    )
    emission = np.asarray(
        [
            [
                [[-0.3, -0.4], [-0.5, -0.2]],
                [[-0.1, -0.7], [-0.4, -0.3]],
                [[-0.8, -0.2], [-0.6, -0.5]],
            ],
            [
                [[-0.6, -0.1], [-0.2, -0.7]],
                [[-0.3, -0.2], [-0.1, -0.4]],
                [[-0.5, -0.6], [-0.3, -0.2]],
            ],
        ],
        dtype=np.float32,
    )
    position_prior = -0.5 * ((positions - 1000.0) / 0.5) ** 2
    rate_prior = -0.5 * (rates / 0.01) ** 2
    bias_prior = -0.5 * (biases / 1.0) ** 2
    mean, _, _, _, kernel_loglik = TRAIN.joint_hmm_forward_backward(
        emission,
        positions,
        rate_log,
        bias_log,
        position_b0,
        position_log,
        reference_log,
        position_prior,
        rate_prior,
        bias_prior,
    )
    transition0, states = dense_transition(
        rate_log, bias_log, position_b0, position_log, reference_log, 0, len(positions)
    )
    transition1, _ = dense_transition(
        rate_log, bias_log, position_b0, position_log, reference_log, 1, len(positions)
    )
    prior = np.asarray(
        [
            math.exp(position_prior[p] + rate_prior[r] + bias_prior[b]) / 2.0
            for p, r, b, _ in states
        ]
    )
    emission0 = np.asarray([math.exp(emission[0, p, b, f]) for p, _, b, f in states])
    emission1 = np.asarray([math.exp(emission[1, p, b, f]) for p, _, b, f in states])
    alpha0 = (prior @ transition0) * emission0
    alpha1 = (alpha0 @ transition1) * emission1
    dense_loglik = math.log(alpha1.sum())
    posterior0 = alpha0 * (transition1 @ emission1)
    posterior1 = alpha1.copy()
    posterior0 /= posterior0.sum()
    posterior1 /= posterior1.sum()
    dense_mean = np.asarray(
        [
            sum(posterior[index] * positions[state[0]] for index, state in enumerate(states))
            for posterior in (posterior0, posterior1)
        ]
    )
    assert np.allclose(mean, dense_mean, atol=2e-6)
    assert np.isclose(kernel_loglik, dense_loglik, atol=2e-5)


def test_local_dtw_reference_transition_is_ordered_and_normalized() -> None:
    config = TRAIN.load_experiment_config()
    outputs = TRAIN.prepare_transition_logs(
        np.asarray([0.0]),
        np.asarray([-0.01, 0.01]),
        np.asarray([-1.0, 1.0]),
        config,
        reference_count=3,
        family="local_dtw_reconstruction",
    )
    reference_log = outputs[-1]
    assert reference_log[0, 2] < -1e17
    assert reference_log[2, 0] < -1e17
    for row in reference_log:
        assert np.isclose(np.exp(row[row > -1e17]).sum(), 1.0)


def test_absolute_tvt_grid_contains_synthetic_unknown_truth() -> None:
    config = TRAIN.load_experiment_config()
    horizontal = synthetic_horizontal(include_truth=True)
    safe = horizontal.drop(columns=["TVT"])
    sequence = TRAIN.sequence_contract(safe, config)
    absolute_tvt, guide, _, pad = TRAIN.build_absolute_tvt_states(
        safe, sequence, synthetic_guide(), config
    )
    truth = horizontal.loc[horizontal["TVT_input"].isna(), "TVT"].to_numpy(np.float64)
    assert np.all(absolute_tvt[-len(truth) :, 0] <= truth)
    assert np.all(absolute_tvt[-len(truth) :, -1] >= truth)
    assert pad == 60.0
    assert np.allclose(guide[-len(truth) :], truth)


def test_reference_family_shapes_and_local_offsets() -> None:
    config = TRAIN.load_experiment_config()
    horizontal = synthetic_horizontal()
    typewell = synthetic_typewell()
    sequence = TRAIN.sequence_contract(horizontal, config)
    absolute_tvt, _, _, _ = TRAIN.build_absolute_tvt_states(
        horizontal, sequence, synthetic_guide(), config
    )
    curve_x = np.arange(980.0, 1040.25, 0.25)
    curve = (curve_x, 100.0 + np.sin((curve_x - 1000.0) / 2.0), np.ones(len(curve_x)))
    atlas = {"coarse": {"g": curve}, "fine": {"g": curve}}
    base = TRAIN.build_reference_tensor(
        horizontal, typewell, sequence, absolute_tvt, "g", atlas, TRAIN.FAMILY_ORDER[0], config
    )
    local = TRAIN.build_reference_tensor(
        horizontal, typewell, sequence, absolute_tvt, "g", atlas, TRAIN.FAMILY_ORDER[1], config
    )
    fine = TRAIN.build_reference_tensor(
        horizontal, typewell, sequence, absolute_tvt, "g", atlas, TRAIN.FAMILY_ORDER[2], config
    )
    assert base.shape == (*absolute_tvt.shape, 2)
    assert local.shape == (*absolute_tvt.shape, 3)
    assert fine.shape == (*absolute_tvt.shape, 1)
    assert not np.allclose(local[:, :, 0], local[:, :, 1])
    assert not np.allclose(local[:, :, 1], local[:, :, 2])


def test_target_free_decoder_rejects_truth_column() -> None:
    config = TRAIN.load_experiment_config()
    with np.testing.assert_raises_regex(ValueError, "forbidden TVT"):
        TRAIN.decode_well_target_free(
            synthetic_horizontal(include_truth=True),
            synthetic_typewell(),
            synthetic_guide(),
            None,
            {"coarse": {}, "fine": {}},
            config,
        )


def test_projection_obeys_absolute_and_integrated_caps() -> None:
    config = TRAIN.load_experiment_config()
    horizontal = synthetic_horizontal()
    eval_index = np.flatnonzero(horizontal["TVT_input"].isna().to_numpy())
    raw = np.full(len(eval_index), 2000.0)
    projected, meta = TRAIN.physical_projection(
        raw, horizontal, eval_index, np.asarray([0.0, 0.01]), config
    )
    assert np.isfinite(projected).all()
    assert meta["integrated_correction_abs_max"] <= 10.0 + 1e-12
    assert -0.25 <= meta["prefix_rate_low"] <= meta["prefix_rate_high"] <= 0.25


def test_source_removed_failed_baseline_and_global_stretch() -> None:
    source = TRAIN_SOURCE.read_text()
    assert "baseline_tvt" not in source
    assert "local_dtw_stretches" not in source
    assert "offset_min_ft" not in source
    assert "local_dtw_reference_offsets_ft" in source
    assert "absolute_tvt_guide" in source
    assert "guide[:, None] + offsets[None, :]" in source


def test_inference_is_blocked_before_hidden_data_access() -> None:
    source = INFERENCE_SOURCE.read_text()
    assert INFERENCE.INFERENCE_BLOCKED is True
    assert "sample_submission.csv" not in source
    assert "pandas" not in source
    with np.testing.assert_raises_regex(RuntimeError, "did not reproduce"):
        INFERENCE.run_inference()
