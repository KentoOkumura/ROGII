from __future__ import annotations

import importlib.util
import inspect
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "experiments" / "exp515_third_place_three_family_hmm_late_submit"
TRAIN_SOURCE = EXP_DIR / "exp515_third_place_three_family_hmm_late_submit_compact_selfcontained_train.py"
INFERENCE_SOURCE = EXP_DIR / "exp515_third_place_three_family_hmm_late_submit_compact_selfcontained_inference.py"


def load_source(path: Path, name: str):
    os.environ["EXP515_IMPORT_ONLY"] = "1"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TRAIN = load_source(TRAIN_SOURCE, "exp515_train_test_module")
INFERENCE = load_source(INFERENCE_SOURCE, "exp515_inference_test_module")


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
    tvt = np.arange(990.0, 1030.5, 0.5)
    return pd.DataFrame({"TVT": tvt, "GR": 100.0 + np.sin((tvt - 1000.0) / 2.0)})


def test_frozen_contract_and_late_submit_identity() -> None:
    config = TRAIN.load_experiment_config()
    TRAIN.validate_frozen_contract(config)
    assert config["experiment"]["route"] == "pf_beam"
    assert config["method_fidelity"]["classification"] == "proxy"
    assert config["late_submission"]["phase"] == "post_competition_late_submission"
    assert "LATE SUBMIT" in config["late_submission"]["message"]
    assert tuple(config["model"]["family_weights"]) == TRAIN.FAMILY_ORDER
    assert math.isclose(sum(config["model"]["family_weights"].values()), 1.0)


def test_nearest_transition_probabilities_are_normalized() -> None:
    logs = TRAIN._nearest_transition_logs(
        np.asarray([1.0, 2.0]),
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
            if rate_value < -1e17 or bias_value < -1e17:
                continue
            center = int(position_b0[time_index, r2])
            position_value = None
            for kernel_index in range(5):
                if p2 == p1 + center - 2 + kernel_index:
                    position_value = position_log[time_index, r2, kernel_index]
                    break
            if position_value is None:
                continue
            transition[source_index, target_index] = math.exp(
                rate_value + bias_value + reference_log[f1, f2] + position_value
            )
    return transition, states


def test_joint_forward_backward_matches_dense_sum_product() -> None:
    config = TRAIN.load_experiment_config()
    dm = np.asarray([1.0, 1.0], dtype=np.float64)
    offsets = np.asarray([-0.5, 0.0, 0.5], dtype=np.float64)
    rates = np.asarray([-0.01, 0.01], dtype=np.float64)
    biases = np.asarray([-1.0, 1.0], dtype=np.float64)
    rate_log, bias_log, position_b0, position_log, reference_log = TRAIN.prepare_transition_logs(
        dm, rates, biases, config, reference_count=2
    )
    emission = np.asarray(
        [
            [[[-0.3, -0.4], [-0.5, -0.2]], [[-0.1, -0.7], [-0.4, -0.3]], [[-0.8, -0.2], [-0.6, -0.5]]],
            [[[-0.6, -0.1], [-0.2, -0.7]], [[-0.3, -0.2], [-0.1, -0.4]], [[-0.5, -0.6], [-0.3, -0.2]]],
        ],
        dtype=np.float32,
    )
    position_prior = -0.5 * (offsets / 0.5) ** 2
    rate_prior = -0.5 * (rates / 0.01) ** 2
    bias_prior = -0.5 * (biases / 1.0) ** 2
    mean, _, _, _, kernel_loglik = TRAIN.joint_hmm_forward_backward(
        emission,
        offsets,
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
        rate_log, bias_log, position_b0, position_log, reference_log, 0, len(offsets)
    )
    transition1, _ = dense_transition(
        rate_log, bias_log, position_b0, position_log, reference_log, 1, len(offsets)
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
    beta1 = np.ones(len(states), dtype=np.float64)
    beta0 = transition1 @ (emission1 * beta1)
    posterior0 = alpha0 * beta0
    posterior1 = alpha1
    posterior0 /= posterior0.sum()
    posterior1 /= posterior1.sum()
    dense_mean = np.asarray(
        [
            sum(posterior[index] * offsets[state[0]] for index, state in enumerate(states))
            for posterior in (posterior0, posterior1)
        ]
    )
    assert np.allclose(mean, dense_mean, atol=2e-6)
    assert np.isclose(kernel_loglik, dense_loglik, atol=2e-5)


def test_fold_atlas_excludes_validation_fold() -> None:
    rows = pd.DataFrame(
        {
            "group": ["g", "g", "g", "g"],
            "source_fold": [0, 0, 1, 1],
            "tvt": [1000.1, 1000.2, 1000.1, 1000.2],
            "gr": [10.0, 10.0, 90.0, 90.0],
        }
    )
    atlas = TRAIN.build_fold_atlases(rows, [0, 1], 0.25, 0.0625)
    assert np.allclose(atlas[0]["coarse"]["g"][1], 90.0)
    assert np.allclose(atlas[1]["coarse"]["g"][1], 10.0)


def test_reference_family_shapes_match_writeup_contract() -> None:
    config = TRAIN.load_experiment_config()
    horizontal = synthetic_horizontal()
    typewell = synthetic_typewell()
    sequence = TRAIN.sequence_contract(horizontal, config)
    offsets = np.arange(-10.0, 10.01, 0.5)
    curve = (
        np.arange(990.0, 1030.25, 0.25),
        np.full(161, 100.0),
        np.ones(161, dtype=np.int32),
    )
    atlas = {"coarse": {"g": curve}, "fine": {"g": curve}}
    assert TRAIN.build_reference_tensor(horizontal, typewell, sequence, offsets, "g", atlas, TRAIN.FAMILY_ORDER[0], config).shape[-1] == 2
    assert TRAIN.build_reference_tensor(horizontal, typewell, sequence, offsets, "g", atlas, TRAIN.FAMILY_ORDER[1], config).shape[-1] == 3
    assert TRAIN.build_reference_tensor(horizontal, typewell, sequence, offsets, "g", atlas, TRAIN.FAMILY_ORDER[2], config).shape[-1] == 1


def test_target_free_decoder_rejects_truth_column() -> None:
    config = TRAIN.load_experiment_config()
    with np.testing.assert_raises_regex(ValueError, "forbidden TVT"):
        TRAIN.decode_well_target_free(
            synthetic_horizontal(include_truth=True), synthetic_typewell(), None, {"coarse": {}, "fine": {}}, config
        )


def test_projection_obeys_absolute_and_integrated_caps() -> None:
    config = TRAIN.load_experiment_config()
    horizontal = synthetic_horizontal()
    eval_index = np.flatnonzero(horizontal["TVT_input"].isna().to_numpy())
    raw = np.full(len(eval_index), 2000.0)
    projected, meta = TRAIN.physical_projection(raw, horizontal, eval_index, np.asarray([0.0, 0.01]), config)
    assert np.isfinite(projected).all()
    assert meta["integrated_correction_abs_max"] <= 10.0 + 1e-12
    assert -0.25 <= meta["prefix_rate_low"] <= meta["prefix_rate_high"] <= 0.25


def test_native_kgram_group_mapping_is_dynamic() -> None:
    typewell = synthetic_typewell()
    values = INFERENCE.quantized_typewell_gr(typewell)
    length = 16
    key = INFERENCE.kgram_key(values[:length])
    matcher = {
        "kgram_index": {key: ("train_a",)},
        "signatures": {"train_a": INFERENCE.normalized_signature(typewell, 128)},
        "kgram_rows": length,
        "query_stride_rows": 4,
        "minimum_matching_kgrams": 1,
        "minimum_fallback_cosine": 0.90,
        "signature_points": 128,
    }
    group, diagnostic = INFERENCE.map_test_group(typewell, matcher, {"train_a": "group_a"})
    assert group == "group_a"
    assert diagnostic["method"] == "native_quantized_gr_kgram"


def test_inference_source_has_no_visible_test_or_lb_rescue_branch() -> None:
    source = INFERENCE_SOURCE.read_text()
    forbidden = ["000d7d20", "00bbac68", "00e12e8b", "14151", "Public LB", "lb_retune"]
    for token in forbidden:
        assert token not in source
    assert "sample_submission.csv" in source
    assert "validate=\"one_to_one\"" in source
    assert "LATE SUBMIT" in source


def test_train_inference_core_formula_parity() -> None:
    shared = [
        "_nearest_transition_logs",
        "prepare_transition_logs",
        "joint_hmm_forward_backward",
        "robust_initial_rate",
        "sequence_contract",
        "robust_sigma_and_bias",
        "build_reference_tensor",
        "build_emission",
        "physical_projection",
        "decode_family",
        "decode_well_target_free",
    ]
    for name in shared:
        assert inspect.getsource(getattr(TRAIN, name)) == inspect.getsource(getattr(INFERENCE, name))
