from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "experiments" / "exp281_exp226_residual_offset_exact_hmm_transition_probe"
TRAIN_SOURCE = EXP_DIR / "exp281_exp226_residual_offset_exact_hmm_transition_probe_train.py"
INFERENCE_SOURCE = EXP_DIR / "exp281_exp226_residual_offset_exact_hmm_transition_probe_inference.py"


def load_module(path: Path = TRAIN_SOURCE, name: str = "exp281_train"):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_config() -> dict:
    value = yaml.safe_load((EXP_DIR / "config.yaml").read_text())
    assert isinstance(value, dict)
    return value


def synthetic_inputs() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    known_tvt = np.array([98.0, 99.0, 100.0, 101.0])
    geop = np.array([109.0, 110.0, 111.0], dtype=float)
    eval_tvt = geop + 10.0
    horizontal = pd.DataFrame(
        {
            "MD": np.arange(7, dtype=float),
            "Z": np.zeros(7, dtype=float),
            "GR": np.r_[known_tvt, eval_tvt],
            "TVT_input": [98.0, 99.0, 100.0, 101.0, np.nan, np.nan, np.nan],
        }
    )
    typewell_tvt = np.linspace(0.0, 200.0, 401)
    typewell = pd.DataFrame({"TVT": typewell_tvt, "GR": typewell_tvt})
    return horizontal, typewell, geop


def test_scientific_contract_is_one_fixed_variant() -> None:
    module = load_module()
    config = load_config()
    module.validate_scientific_contract(config)
    config["model"]["hmm"]["delta_max_ft"] = 81.0
    with pytest.raises(ValueError, match="contract changed"):
        module.validate_scientific_contract(config)


def test_row_dependent_gr_emission_peaks_at_positive_ten_offset() -> None:
    module = load_module(name="exp281_emission")
    horizontal, typewell, geop = synthetic_inputs()
    prepared = module.prepare_hmm_inputs(horizontal, typewell, geop, load_config())
    grid = prepared["grid"]
    peak_delta = grid[np.argmax(prepared["emission_ll"], axis=1)]
    assert np.all(np.abs(peak_delta - 10.0) <= 0.35)
    assert prepared["delta_grid_coverage_rows"] == len(geop)
    assert prepared["delta_grid_rows"] == len(geop)
    assert prepared["r0"] == 0.0
    result = module.run_residual_offset_hmm(horizontal, typewell, geop, load_config())
    assert result["mean"].shape == geop.shape
    assert np.isfinite(result["mean"]).all()
    assert np.isfinite(result["std"]).all()
    assert np.isfinite(result["loglik"])


def test_offset_transition_center_excludes_absolute_z_motion() -> None:
    source = TRAIN_SOURCE.read_text()
    assert "mu = rates[r2] * dm[t_i] - dz[t_i]" not in source
    assert source.count("mu = rates[r2] * dm[t_i]") == 2
    config = load_config()
    assert config["model"]["hmm"]["transition_center"] == "exp226_tvt_geop_row_delta"


def test_decoder_preparation_and_raw_reader_exclude_suffix_truth(tmp_path: Path) -> None:
    module = load_module(name="exp281_leakage")
    horizontal, typewell, geop = synthetic_inputs()
    horizontal["TVT"] = np.arange(len(horizontal), dtype=float)
    with pytest.raises(ValueError, match="forbids unknown-suffix true TVT"):
        module.prepare_hmm_inputs(horizontal, typewell, geop, load_config())
    horizontal.to_csv(tmp_path / "well_a__horizontal_well.csv", index=False)
    typewell.to_csv(tmp_path / "well_a__typewell.csv", index=False)
    safe_horizontal, loaded_typewell = module.load_well("well_a", tmp_path)
    assert "TVT" not in safe_horizontal.columns
    assert "TVT" in loaded_typewell.columns


def test_persistent_offset_recovery_is_truth_only_posthoc() -> None:
    module = load_module(name="exp281_recovery")
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
            "residual_offset_hmm": np.r_[np.full(128, 11.0), np.zeros(rows - 128)],
        }
    )
    episodes, summary = module.persistent_offset_episodes(frame, load_config())
    candidate = episodes.loc[episodes["candidate"] == "residual_offset_hmm"].iloc[0]
    assert candidate["consecutive_rows_above_threshold"] == 128
    assert candidate["recovery_rows_after_confirmation"] == 1
    assert bool(candidate["recovered_within_256"])
    candidate_summary = summary.loc[summary["candidate"] == "residual_offset_hmm"].iloc[0]
    assert candidate_summary["episodes"] == 1
    assert candidate_summary["recovered_within_512_rate"] == 1.0


def test_inference_contract_is_fail_closed() -> None:
    module = load_module(INFERENCE_SOURCE, "exp281_inference")
    config = load_config()
    contract = module.validate_disabled_inference(config)
    assert not contract["inference_enabled"]
    assert not contract["create_submission"]
    config["inference"]["enabled"] = True
    with pytest.raises(ValueError, match="must remain disabled"):
        module.validate_disabled_inference(config)
