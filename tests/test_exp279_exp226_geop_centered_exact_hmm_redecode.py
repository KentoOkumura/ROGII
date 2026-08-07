from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "experiments" / "exp279_exp226_geop_centered_exact_hmm_redecode"
SOURCE = EXP_DIR / "exp279_exp226_geop_centered_exact_hmm_redecode_train.py"


def load_module():
    spec = importlib.util.spec_from_file_location("exp279_train", SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_config() -> dict:
    value = yaml.safe_load((EXP_DIR / "config.yaml").read_text())
    assert isinstance(value, dict)
    return value


def synthetic_inputs() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    horizontal = pd.DataFrame(
        {
            "MD": np.arange(7, dtype=float),
            "Z": np.zeros(7, dtype=float),
            "GR": np.full(7, 50.0),
            "TVT_input": [98.0, 99.0, 100.0, 101.0, np.nan, np.nan, np.nan],
        }
    )
    typewell = pd.DataFrame(
        {
            "TVT": np.linspace(0.0, 200.0, 401),
            "GR": np.full(401, 50.0),
        }
    )
    geop = np.array([109.0, 110.0, 111.0], dtype=float)
    return horizontal, typewell, geop


def test_scientific_contract_is_one_fixed_variant() -> None:
    module = load_module()
    config = load_config()
    module.validate_scientific_contract(config)
    config["model"]["geometry_unary"]["lambda"] = 0.51
    with pytest.raises(ValueError, match="sigma20/lambda0.50"):
        module.validate_scientific_contract(config)


def test_geometry_unary_peaks_at_each_exp226_center() -> None:
    module = load_module()
    horizontal, typewell, geop = synthetic_inputs()
    prepared = module.prepare_hmm_inputs(horizontal, typewell, geop, load_config())
    grid = prepared["grid"]
    peak = grid[np.argmax(prepared["emission_ll"], axis=1)]
    assert np.all(np.abs(peak - geop) <= 0.35)
    assert prepared["geop_inside_grid_rows"] == len(geop)
    assert prepared["geop_rows"] == len(geop)


def test_decoder_preparation_rejects_unknown_suffix_truth() -> None:
    module = load_module()
    horizontal, typewell, geop = synthetic_inputs()
    horizontal["TVT"] = np.arange(len(horizontal), dtype=float)
    with pytest.raises(ValueError, match="forbids unknown-suffix true TVT"):
        module.prepare_hmm_inputs(horizontal, typewell, geop, load_config())


def test_persistent_offset_recovery_is_truth_only_posthoc() -> None:
    module = load_module()
    rows = 160
    frame = pd.DataFrame(
        {
            "well": "well_a",
            "row_idx": np.arange(rows),
            "fold": 0,
            "true_tvt_readout_only": np.zeros(rows),
            "exp226_pred": np.zeros(rows),
            "exact_hmm": np.zeros(rows),
            "exp263_fixed": np.zeros(rows),
            "geop_hmm": np.r_[np.full(128, 11.0), np.zeros(rows - 128)],
        }
    )
    episodes, summary = module.persistent_offset_episodes(frame, load_config())
    geop = episodes.loc[episodes["candidate"] == "geop_hmm"].iloc[0]
    assert geop["consecutive_rows_above_threshold"] == 128
    assert geop["recovery_rows_after_confirmation"] == 1
    assert bool(geop["recovered_within_256"])
    geop_summary = summary.loc[summary["candidate"] == "geop_hmm"].iloc[0]
    assert geop_summary["episodes"] == 1
    assert geop_summary["recovered_within_512_rate"] == 1.0
