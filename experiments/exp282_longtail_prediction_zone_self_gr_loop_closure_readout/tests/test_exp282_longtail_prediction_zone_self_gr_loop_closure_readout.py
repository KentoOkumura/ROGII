from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = ROOT / "experiments" / "exp282_longtail_prediction_zone_self_gr_loop_closure_readout"
TRAIN_SOURCE = (
    EXP_DIR
    / "exp282_longtail_prediction_zone_self_gr_loop_closure_readout_compact_selfcontained_train.py"
)
INFERENCE_SOURCE = (
    EXP_DIR
    / (
        "exp282_longtail_prediction_zone_self_gr_loop_closure_readout_"
        "compact_selfcontained_inference.py"
    )
)


def load_module(path: Path = TRAIN_SOURCE, name: str = "exp282_train"):
    previous = os.environ.get("EXP282_IMPORT_ONLY")
    os.environ["EXP282_IMPORT_ONLY"] = "1"
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if previous is None:
            os.environ.pop("EXP282_IMPORT_ONLY", None)
        else:
            os.environ["EXP282_IMPORT_ONLY"] = previous


def test_config_and_zero_booster_contract() -> None:
    module = load_module(name="exp282_contract")
    config = yaml.safe_load((EXP_DIR / "config.yaml").read_text())
    module.validate_scientific_contract(config)
    assert config["experiment"]["route"] == "pf_beam"
    assert config["execution"]["active_variants"] == 1
    assert config["execution"]["lightgbm_config_count"] == 0
    assert config["execution"]["trained_fold_count"] == 0
    assert config["execution"]["total_boosters"] == 0
    assert config["execution"]["hmm_variants"] == 0
    assert config["execution"]["pf_variants"] == 0
    assert config["execution"]["control_or_parent_retraining"] is False
    assert config["execution"]["kaggle_push_approved"] is True


def test_forward_reverse_and_tie_break_matching() -> None:
    module = load_module(name="exp282_matching")
    rng = np.random.default_rng(123)
    signal = rng.normal(0.0, 0.01, size=190)
    motif_a = np.asarray([-2.0, -1.0, 0.0, 0.5, 2.0, 1.0, 0.0, -0.5, -1.5, 0.2, 1.3])
    motif_b = np.asarray([0.0, 1.5, -0.5, 2.0, 0.3, -1.0, -2.0, 0.8, 1.1, -0.2, 0.4])
    signal[25:36] = motif_a
    signal[65:76] = motif_b
    signal[115:126] = motif_a
    signal[145:156] = motif_b[::-1]
    matched = module.match_one_scale(
        signal,
        np.asarray([30, 70]),
        np.asarray([120, 150]),
        half_window=5,
        chunk_size=1,
    )
    assert matched["donor_row_idx"].tolist() == [30, 70]
    assert matched["orientation"].tolist() == ["forward", "reverse"]
    assert (matched["best_ncc"] > 0.999).all()

    tie_signal = rng.normal(0.0, 0.01, size=110)
    tie_signal[15:26] = motif_a
    tie_signal[35:46] = motif_a
    tie_signal[75:86] = motif_a
    tied = module.match_one_scale(
        tie_signal,
        np.asarray([20, 40]),
        np.asarray([80]),
        half_window=5,
        chunk_size=8,
    )
    assert int(tied.loc[0, "donor_row_idx"]) == 20
    assert tied.loc[0, "orientation"] == "forward"


def test_segment_support_forward_reverse_and_break() -> None:
    module = load_module(name="exp282_segment")
    edges = pd.DataFrame(
        {
            "well": ["a"] * 7,
            "receiver_row_idx": [100, 103, 106, 109, 112, 115, 118],
            "donor_row_idx": [30, 33, 36, 70, 67, 64, 20],
            "orientation": [
                "forward",
                "forward",
                "forward",
                "reverse",
                "reverse",
                "reverse",
                "forward",
            ],
        }
    )
    supported = module.add_segment_support(edges, stride=3, minimum_centers=3)
    assert supported["segment_run_length"].tolist() == [3, 3, 3, 3, 3, 3, 1]
    assert supported["segment_supported"].tolist() == [True] * 6 + [False]
    assert supported["orientation_flip_count"].nunique() == 1
    assert int(supported["orientation_flip_count"].iloc[0]) == 2


def test_stable_shuffle_is_well_local_and_deterministic() -> None:
    module = load_module(name="exp282_shuffle")
    edges = pd.DataFrame(
        {
            "well": ["a"] * 8 + ["b"] * 8,
            "receiver_row_idx": list(range(8)) + list(range(8)),
            "donor_row_idx": list(range(10, 18)) + list(range(30, 38)),
            "edge_confidence": np.linspace(0.0, 1.0, 16),
        }
    )
    first = module.add_stable_shuffled_control(
        edges, experiment_name=module.EXPERIMENT_NAME, seed=42
    )
    second = module.add_stable_shuffled_control(
        edges, experiment_name=module.EXPERIMENT_NAME, seed=42
    )
    assert first["shuffled_donor_row_idx"].equals(second["shuffled_donor_row_idx"])
    assert first["receiver_row_idx"].equals(edges["receiver_row_idx"])
    assert first["edge_confidence"].equals(edges["edge_confidence"])
    for well in ("a", "b"):
        original = sorted(edges.loc[edges["well"] == well, "donor_row_idx"].tolist())
        shuffled = sorted(first.loc[first["well"] == well, "shuffled_donor_row_idx"].tolist())
        assert shuffled == original


def test_truth_and_forbidden_columns_fail_closed(tmp_path: Path) -> None:
    module = load_module(name="exp282_leakage")
    safe = pd.DataFrame(
        {
            "MD": [0.0, 1.0],
            "GR": [10.0, 11.0],
            "TVT_input": [100.0, np.nan],
        }
    )
    module.validate_score_stage_columns(safe)
    with pytest.raises(ValueError, match="forbidden"):
        module.validate_score_stage_columns(safe.assign(TVT=[100.0, 101.0]))
    keys = pd.DataFrame({"well": ["a"], "well_row_idx": [1]})
    with pytest.raises(ValueError, match="edge content SHA"):
        module.load_truth_for_keys(tmp_path, keys, frozen_edge_sha256="")


def test_inference_is_fail_closed() -> None:
    module = load_module(INFERENCE_SOURCE, "exp282_inference")
    config = yaml.safe_load((EXP_DIR / "config.yaml").read_text())
    module.assert_inference_disabled(config)
    with pytest.raises(RuntimeError, match="train-side zero-booster readout"):
        module.fail_closed()
