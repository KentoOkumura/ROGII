from __future__ import annotations

import inspect
import os
import runpy
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "experiments" / "exp273_two_dimensional_formation_gradient_transition"
TRAIN = EXP_DIR / "exp273_two_dimensional_formation_gradient_transition_train.py"


def load_namespace() -> dict[str, object]:
    previous = os.environ.get("EXP273_IMPORT_ONLY")
    os.environ["EXP273_IMPORT_ONLY"] = "1"
    try:
        return runpy.run_path(str(TRAIN))
    finally:
        if previous is None:
            os.environ.pop("EXP273_IMPORT_ONLY", None)
        else:
            os.environ["EXP273_IMPORT_ONLY"] = previous


def plane_config() -> dict[str, object]:
    config = yaml.safe_load((EXP_DIR / "config.yaml").read_text())
    return config["model"]["formation_plane"]


def test_huber_plane_recovers_two_dimensional_gradient_and_fixed_prototypes() -> None:
    namespace = load_namespace()
    fit_formation_plane = namespace["fit_formation_plane"]
    theta = np.linspace(0.0, 2.0 * np.pi, 240, endpoint=False)
    x = 120.0 * np.cos(theta) + 0.2 * np.arange(len(theta))
    y = 70.0 * np.sin(theta)
    z = np.linspace(1000.0, 1040.0, len(theta))
    surface = 0.035 * x - 0.018 * y + 250.0
    surface[[20, 80, 160]] += np.array([80.0, -60.0, 100.0])
    prefix = pd.DataFrame(
        {"X": x, "Y": y, "Z": z, "TVT_input": surface - z}
    )
    result = fit_formation_plane(prefix, plane_config())
    assert result["valid"] is True
    assert result["fallback_reason"] == "ok"
    assert abs(result["gradient_x"] - 0.035) < 0.002
    assert abs(result["gradient_y"] + 0.018) < 0.002
    prototypes = result["prototypes"]
    assert tuple(prototypes) == (
        "center",
        "axis1_minus",
        "axis1_plus",
        "axis2_minus",
        "axis2_plus",
    )
    center = np.asarray(prototypes["center"])
    assert np.allclose(
        np.asarray(prototypes["axis1_minus"])
        + np.asarray(prototypes["axis1_plus"]),
        2.0 * center,
    )
    assert np.allclose(
        np.asarray(prototypes["axis2_minus"])
        + np.asarray(prototypes["axis2_plus"]),
        2.0 * center,
    )


def test_straight_xy_prefix_fails_closed_to_zero_gradient() -> None:
    namespace = load_namespace()
    fit_formation_plane = namespace["fit_formation_plane"]
    x = np.arange(128, dtype=np.float64)
    y = np.zeros_like(x)
    z = 1000.0 + 0.1 * x
    surface = 0.02 * x + 100.0
    prefix = pd.DataFrame(
        {"X": x, "Y": y, "Z": z, "TVT_input": surface - z}
    )
    result = fit_formation_plane(prefix, plane_config())
    assert result["valid"] is False
    assert "xy_rank_below_2" in result["fallback_reason"]
    for gradient in result["prototypes"].values():
        assert np.array_equal(np.asarray(gradient), np.zeros(2))


def test_residual_initial_rate_subtracts_fixed_gradient_projection() -> None:
    namespace = load_namespace()
    residual_initial_rate = namespace["residual_initial_rate"]
    rows = 80
    md = np.arange(rows, dtype=np.float64)
    x = 2.0 * md
    y = 0.5 * md + 4.0 * np.sin(md / 8.0)
    z = 1000.0 + 0.3 * md
    gx, gy, residual_rate = 0.04, -0.015, 0.012
    surface = gx * x + gy * y + residual_rate * md + 300.0
    prefix = pd.DataFrame(
        {"MD": md, "X": x, "Y": y, "Z": z, "TVT_input": surface - z}
    )
    rate, effective_rows, valid_steps = residual_initial_rate(
        prefix, gx, gy, window_rows=30
    )
    assert abs(rate - residual_rate) < 1.0e-12
    assert effective_rows == 30
    assert valid_steps == 29


def test_target_free_stable_two_shard_partition() -> None:
    namespace = load_namespace()
    stable_well_shard = namespace["stable_well_shard"]
    assignments = [stable_well_shard(f"well_{index:04d}", 2) for index in range(200)]
    assert set(assignments) == {0, 1}
    assert assignments == [
        stable_well_shard(f"well_{index:04d}", 2) for index in range(200)
    ]


def test_config_freezes_five_prototypes_and_zero_booster_contract() -> None:
    namespace = load_namespace()
    validate_scientific_contract = namespace["validate_scientific_contract"]
    config = yaml.safe_load((EXP_DIR / "config.yaml").read_text())
    validate_scientific_contract(config)
    assert config["experiment"]["route"] == "pf_beam"
    assert config["model"]["formation_plane"]["candidate_names"] == [
        "center",
        "axis1_minus",
        "axis1_plus",
        "axis2_minus",
        "axis2_plus",
    ]
    assert config["model"]["control"]["regenerate"] is False
    assert config["execution"]["active_hmm_variants"] == 5
    assert config["execution"]["shard_count"] == 2
    assert config["execution"]["max_hmm_well_runs"] == 3865
    assert config["execution"]["lightgbm_config_count"] == 0
    assert config["execution"]["fold_count"] == 0
    assert config["execution"]["total_boosters"] == 0
    assert config["execution"]["kaggle_push_approved"] is True
    assert [item["expected_rows"] for item in config["data"]["shard_outputs"]] == [
        1910995,
        1872994,
    ]
    assert all(
        len(item["expected_decompressed_sha256"]) == 64
        for item in config["data"]["shard_outputs"]
    )
    assert config["audit"]["persist_candidate_mean"] is False
    assert config["audit"]["persist_oracle_predictions"] is False
    assert config["audit"]["fallback_control_parity_atol_ft"] == 0.00001
    assert config["audit"]["turning_azimuth_coverage"] == 0.10
    assert config["inference"]["enabled"] is False


def test_generator_drops_target_before_every_hmm_call_and_attaches_it_after() -> None:
    namespace = load_namespace()
    source = inspect.getsource(namespace["build_gradient_rows_for_well"])
    drop_position = source.index('generation_horizontal = horizontal.drop(columns=["TVT"]).copy()')
    hmm_position = source.index("run_hmm2(\n                generation_horizontal")
    target_position = source.index("true_tvt =")
    assert drop_position < hmm_position < target_position
    assert "run_hmm2(\n                horizontal" not in source
    assert 'fit_formation_plane(known, get_nested(config, "model.formation_plane")' in source


def test_hmm_transition_adds_gradient_surface_move_without_changing_emission() -> None:
    namespace = load_namespace()
    kernel_source = inspect.getsource(namespace["_hmm2_fb"])
    run_source = inspect.getsource(namespace["run_hmm2"])
    assert kernel_source.count(
        "mu = surface_move[t_i] + rates[r2] * dm[t_i] - dz[t_i]"
    ) == 2
    assert "surface_move = float(gradient_x) * dx + float(gradient_y) * dy" in run_source
    assert "zscore = (gr[:, None] - gr_grid[None, :]) / gr_sigma" in run_source


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
        "exp273_two_dimensional_formation_gradient_transition_train.py": "aggregate",
        "exp273_two_dimensional_formation_gradient_transition_train_variant0.py": "shard0",
        "exp273_two_dimensional_formation_gradient_transition_train_variant1.py": "shard1",
    }
    for filename, mode in expected_modes.items():
        source = (EXP_DIR / filename).read_text()
        assert f'RUN_KIND_OVERRIDE = "{mode}"' in source
        assert "from settings import" not in source
        assert "__file__" not in source
        assert "# ## Contents" in source
        assert '"candidate_mean_persisted": False' in source
        assert '"oracle_prediction_persisted": False' in source
        assert "EXP268_IMPORT_ONLY" not in source
    inference_source = (
        EXP_DIR / "exp273_two_dimensional_formation_gradient_transition_inference.py"
    ).read_text()
    assert '"submission_creation": False' in inference_source
    assert '"gradient_hard_switch": False' in inference_source
    assert "raise RuntimeError" in inference_source
    assert "__file__" not in inference_source
