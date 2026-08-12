from __future__ import annotations

import inspect
import os
import runpy
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = ROOT / "experiments" / "exp268_multi_scale_initial_rate_candidates"
TRAIN = EXP_DIR / "exp268_multi_scale_initial_rate_candidates_train.py"


def load_namespace() -> dict[str, object]:
    previous = os.environ.get("EXP268_IMPORT_ONLY")
    os.environ["EXP268_IMPORT_ONLY"] = "1"
    try:
        return runpy.run_path(str(TRAIN))
    finally:
        if previous is None:
            os.environ.pop("EXP268_IMPORT_ONLY", None)
        else:
            os.environ["EXP268_IMPORT_ONLY"] = previous


def test_fixed_windows_and_known_prefix_rate_only() -> None:
    namespace = load_namespace()
    robust_initial_rate = namespace["robust_initial_rate"]
    rows = 300
    md = np.arange(rows, dtype=np.float64)
    surface_step = np.where(md < 180, 0.01, np.where(md < 250, 0.03, -0.02))
    surface = np.cumsum(surface_step)
    z = np.linspace(0.0, 3.0, rows)
    prefix = pd.DataFrame({"MD": md, "Z": z, "TVT_input": surface - z})
    rates = {}
    for window in (32, 64, 128, 256):
        rate, effective_rows, valid_steps = robust_initial_rate(prefix, window)
        tail = prefix.tail(window)
        expected = np.median(
            (np.diff(tail["TVT_input"]) + np.diff(tail["Z"])) / np.diff(tail["MD"])
        )
        assert rate == expected
        assert effective_rows == window
        assert valid_steps == window - 1
        rates[window] = rate
    assert len(set(rates.values())) >= 3


def test_target_free_stable_two_shard_partition() -> None:
    namespace = load_namespace()
    stable_well_shard = namespace["stable_well_shard"]
    assignments = [stable_well_shard(f"well_{index:04d}", 2) for index in range(200)]
    assert set(assignments) == {0, 1}
    assert assignments == [
        stable_well_shard(f"well_{index:04d}", 2) for index in range(200)
    ]


def test_config_freezes_candidate_bank_and_zero_booster_contract() -> None:
    namespace = load_namespace()
    validate_scientific_contract = namespace["validate_scientific_contract"]
    config = yaml.safe_load((EXP_DIR / "config.yaml").read_text())
    validate_scientific_contract(config)
    assert config["experiment"]["route"] == "pf_beam"
    assert config["model"]["initial_rate"]["windows"] == [32, 64, 128, 256]
    assert config["model"]["control"]["rate_window_rows"] == 30
    assert config["model"]["control"]["regenerate"] is False
    assert config["execution"]["active_hmm_variants"] == 4
    assert config["execution"]["shard_count"] == 2
    assert config["execution"]["lightgbm_config_count"] == 0
    assert config["execution"]["fold_count"] == 0
    assert config["execution"]["total_boosters"] == 0
    assert config["audit"]["persist_candidate_mean"] is False
    assert config["audit"]["persist_oracle_predictions"] is False
    assert config["inference"]["enabled"] is False


def test_generator_drops_target_before_every_hmm_call() -> None:
    namespace = load_namespace()
    source = inspect.getsource(namespace["build_multiscale_rows_for_well"])
    drop_position = source.index('generation_horizontal = horizontal.drop(columns=["TVT"])')
    hmm_position = source.index("run_hmm2(\n            generation_horizontal")
    target_position = source.index("true_tvt =")
    assert drop_position < hmm_position < target_position
    assert "run_hmm2(\n            horizontal" not in source


def test_oracle_scope_semantics_are_diagnostic_only() -> None:
    namespace = load_namespace()
    oracle_prediction = namespace["oracle_prediction"]
    frame = pd.DataFrame(
        {
            "well": ["w"] * 4,
            "row_idx": [0, 1, 2, 3],
            "true_tvt": [0.0, 0.0, 10.0, 10.0],
            "a": [0.0, 0.0, 0.0, 0.0],
            "b": [10.0, 10.0, 10.0, 10.0],
        }
    )
    row = oracle_prediction(frame, ["a", "b"], "row")
    block = oracle_prediction(frame, ["a", "b"], "block", block_rows=2)
    whole = oracle_prediction(frame, ["a", "b"], "whole_well")
    assert np.array_equal(row, np.array([0.0, 0.0, 10.0, 10.0], dtype=np.float32))
    assert np.array_equal(block, row)
    assert np.array_equal(whole, np.zeros(4, dtype=np.float32))


def test_all_notebook_sources_are_self_contained_and_mode_pinned() -> None:
    expected_modes = {
        "exp268_multi_scale_initial_rate_candidates_train.py": "aggregate",
        "exp268_multi_scale_initial_rate_candidates_train_variant0.py": "shard0",
        "exp268_multi_scale_initial_rate_candidates_train_variant1.py": "shard1",
    }
    for filename, mode in expected_modes.items():
        source = (EXP_DIR / filename).read_text()
        assert f'RUN_KIND_OVERRIDE = "{mode}"' in source
        assert "from settings import" not in source
        assert "__file__" not in source
        assert "# ## Contents" in source
        assert "candidate_mean_persisted" in source
        assert '"candidate_mean_persisted": False' in source
        assert '"oracle_prediction_persisted": False' in source
    inference_source = (EXP_DIR / "exp268_multi_scale_initial_rate_candidates_inference.py").read_text()
    assert "submission_creation\": False" in inference_source
    assert "raise RuntimeError" in inference_source
    assert "__file__" not in inference_source
