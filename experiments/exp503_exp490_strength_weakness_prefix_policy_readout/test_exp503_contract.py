from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).parent
SOURCE = HERE / "exp503_exp490_strength_weakness_prefix_policy_readout_compact_selfcontained_train.py"


def load_module():
    spec = importlib.util.spec_from_file_location("exp503_train", SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_config_and_execution_contract() -> None:
    module = load_module()
    config = module.load_config(HERE / "config.yaml")
    profiles = module.fade_profiles(config)
    assert len(profiles) == 29
    assert sum(profile["profile"] == "alpha1_tau85" for profile in profiles) == 1
    assert sum(profile["profile"] == "alpha1_tau500" for profile in profiles) == 1
    assert config["execution_contract"]["gpu_runs"] == 0
    assert config["execution_contract"]["new_hmm_well_runs"] == 0
    assert config["implementation"]["inference_enabled"] is False


def test_public_fade_formula() -> None:
    module = load_module()
    md = np.array([0.0, 85.0, 170.0])
    ramp = module.fade_ramp(md, 85.0)
    assert np.allclose(ramp, [0.0, 1.0 - np.exp(-1.0), 1.0 - np.exp(-2.0)])
    assert np.allclose(module.fade_ramp(md, 0.0), 1.0)


def test_outer_profile_selection_never_reads_held_sse() -> None:
    module = load_module()
    arrays = {
        "fold": np.array([0, 0, 1, 1], dtype=np.int8),
        "md_since": np.ones(4),
    }
    profiles = [
        {"profile": "parent", "alpha": 0.0, "tau_ft": 0.0},
        {"profile": "candidate", "alpha": 1.0, "tau_ft": 0.0},
    ]
    fold_sse = np.array([[100.0, 1.0], [1.0, 100.0]])
    weights, selections = module.outer_global_fade_policy(
        arrays, profiles, fold_sse, np.array([0, 1])
    )
    assert selections[0]["selected_profile"] == "parent"
    assert selections[1]["selected_profile"] == "candidate"
    assert np.allclose(weights, [0.0, 0.0, 1.0, 1.0])


def test_source_contains_no_upstream_candidate_generation() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    forbidden_calls = ("run_particle_filter(", "run_pf(", "run_beam(", "LGBMRegressor(")
    assert all(call not in source for call in forbidden_calls)


def test_well_readout_preserves_single_rows_feature() -> None:
    module = load_module()
    arrays = {
        "wells": np.array(["a", "b"]),
        "well_codes": np.array([0, 0, 1, 1], dtype=np.int32),
        "n_wells": 2,
        "fold": np.array([0, 0, 1, 1], dtype=np.int8),
        "suffix_offset": np.array([0.0, 1.0, 0.0, 1.0]),
        "truth": np.array([0.0, 1.0, 0.0, 1.0]),
        "parent": np.array([1.0, 2.0, 1.0, 2.0]),
        "candidate": np.array([0.5, 1.5, 0.5, 1.5]),
        "exp226": np.array([0.0, 1.0, 0.0, 1.0]),
        "geometry": np.array([0.0, 1.0, 0.0, 1.0]),
        "parent_error": np.ones(4),
        "candidate_error": np.full(4, 0.5),
        "delta": np.full(4, -0.5),
    }
    features = pd.DataFrame({"well": ["a", "b"], "rows": [2, 2], "prefix_x": [1.0, 2.0]})
    result = module.build_well_readout(arrays, features, window=2)
    assert list(result.columns).count("rows") == 1
    assert result["rows"].tolist() == [2, 2]
    assert result["prefix_x"].tolist() == [1.0, 2.0]
