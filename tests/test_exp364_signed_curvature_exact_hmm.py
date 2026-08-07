from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "experiments" / "exp364_signed_curvature_exact_hmm"
TRAIN_SOURCE = (
    EXP_DIR
    / "exp364_signed_curvature_exact_hmm_compact_selfcontained_train.py"
)
INFERENCE_SOURCE = (
    EXP_DIR
    / "exp364_signed_curvature_exact_hmm_compact_selfcontained_inference.py"
)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def train():
    return load_module(TRAIN_SOURCE, "exp364_train_test")


@pytest.fixture(scope="module")
def config():
    value = yaml.safe_load((EXP_DIR / "config.yaml").read_text())
    assert isinstance(value, dict)
    return value


def test_frozen_contract_is_stage0_only_and_stage1_fail_closed(train, config):
    contract = train.validate_scientific_contract(config)
    assert contract["candidate_signs"] == [-1, 0, 1]
    assert contract["resource_projection_wells"] == 16
    assert contract["hmm_well_runs"] == 0
    assert contract["run_stage_1"] is False
    assert len(contract["contract_sha256"]) == 64
    approved = train.validate_scientific_contract(
        config,
        require_run_approval=True,
    )
    assert approved["contract_sha256"] == contract["contract_sha256"]
    assert config["execution"]["run_stage_0"] is True
    assert config["execution"]["run_inference"] is False
    assert config["execution"]["create_submission"] is False


def test_curvature_transition_and_rate_drift_are_exactly_frozen(train, config):
    transition = np.asarray(
        config["model"]["signed_curvature"]["transition_matrix"],
        dtype=np.float64,
    )
    expected = np.asarray(
        [
            [511.0 / 512.0, 1.0 / 512.0, 0.0],
            [1.0 / 2048.0, 1023.0 / 1024.0, 1.0 / 2048.0],
            [0.0, 1.0 / 512.0, 511.0 / 512.0],
        ]
    )
    np.testing.assert_array_equal(transition, expected)
    assert config["model"]["signed_curvature"]["rate_drift_per_row"] == pytest.approx(
        0.005 / 512.0
    )


def test_fixed_signed_paths_preserve_sign_order_and_zero_path(train):
    rows = 600
    paths = train.generate_fixed_signed_paths(
        eval_md=np.arange(1.0, rows + 1.0),
        eval_z=np.zeros(rows),
        last_known_md=0.0,
        last_known_position=1000.0,
        initial_rate=0.0,
        signs=[-1, 0, 1],
        momentum=0.998,
        rate_drift_per_row=0.000009765625,
        grid_min=0.0,
        grid_max=2000.0,
    )
    assert paths[-1][-1] < paths[0][-1] < paths[1][-1]
    np.testing.assert_allclose(paths[0], 1000.0)
    assert np.isfinite(np.stack(list(paths.values()))).all()


def test_exp209_grid_and_block_contract(train):
    grid_min, grid_max, count = train.exp209_position_grid_bounds(
        last_known_tvt=1000.0,
        typewell_tvt=np.asarray([700.0, 1000.0, 1300.0]),
        band_pad_ft=100.0,
        typewell_outer_pad_ft=40.0,
        step_ft=0.35,
    )
    assert grid_min == pytest.approx(900.0)
    expected_grid = np.arange(900.0, 1100.0 + 0.35, 0.35)
    assert grid_max == pytest.approx(expected_grid[-1])
    assert count == len(expected_grid)
    assert train.fixed_full_blocks(1024, 512, 256) == [
        (0, 512),
        (256, 768),
        (512, 1024),
    ]


def test_circular_control_and_tie_break_are_deterministic(train):
    blocks = [np.full(512, value, dtype=float) for value in range(3)]
    shifted = train.circular_control_gr_blocks(blocks, 1, 256)
    assert [float(block[0]) for block in shifted] == [1.0, 2.0, 0.0]
    ranking = train.ranked_signs(
        {-1: -1.0, 0: -1.0, 1: -1.0},
        tie_break=[0, -1, 1],
    )
    assert ranking == [0, -1, 1]


def synthetic_resource_manifest() -> pd.DataFrame:
    rows = []
    for index in range(30):
        suffix_rows = 512 + index * 32
        position_count = 200 + index
        rows.append(
            {
                "well_id": f"well_{index:02d}",
                "status": "ok",
                "prefix_rows": 300,
                "suffix_rows": suffix_rows,
                "position_grid_count": position_count,
                "rate_grid_count": 41,
                "parent_state_cell_rows": suffix_rows * position_count * 41,
            }
        )
    return pd.DataFrame(rows)


def test_resource_projection_selects_16_quantiles_including_extrema(
    train, config
):
    manifest = synthetic_resource_manifest()
    selected, summary = train.build_resource_projection(manifest, config)
    assert len(selected) == 16
    assert selected["well_id"].iloc[0] == "well_00"
    assert selected["well_id"].iloc[-1] == "well_29"
    assert summary["includes_minimum_workload"]
    assert summary["includes_maximum_workload"]
    assert summary["scientific_hmm_well_runs"] == 0
    assert summary["projected_runtime_seconds"] == pytest.approx(
        11285.868 * 3.0
    )
    assert summary["projected_peak_rss_gb"] > 0.0


def test_truth_access_ledger_rejects_prefreeze_target(train):
    ledger = train.TruthAccessLedger()
    with pytest.raises(ValueError, match="forbidden pre-freeze"):
        ledger.guard_prefreeze_columns(["MD", "TVT"], 12, "synthetic")
    assert ledger.truth_rows_before_freeze == 12


def test_ranking_metrics_reports_positive_zero_path_gain(train):
    frame = pd.DataFrame(
        {
            "well_id": ["a", "b"],
            "block_rows": [512, 512],
            "real_top1_correct": [True, True],
            "zero_top1_correct": [False, True],
            "circular_top1_correct": [False, True],
            "real_reciprocal_rank": [1.0, 1.0],
            "zero_first_reciprocal_rank": [0.5, 1.0],
            "circular_reciprocal_rank": [0.5, 1.0],
            "selected_mse": [4.0, 9.0],
            "zero_mse": [16.0, 9.0],
        }
    )
    metrics = train.ranking_metrics(frame)
    assert metrics["top1_gain_vs_zero_first"] == pytest.approx(0.5)
    assert metrics["mrr_gain_vs_zero_first"] == pytest.approx(0.25)
    assert metrics["selected_path_rmse_gain_vs_zero_ft"] > 0.0


def test_inference_is_fail_closed_and_sources_are_not_thin(config):
    inference = load_module(INFERENCE_SOURCE, "exp364_inference_test")
    counts = inference.validate_disabled_inference(config)
    assert counts["fixed_signed_paths"] == 3
    assert counts["exact_hmm_well_runs"] == 0
    with pytest.raises(RuntimeError, match="Stage 1 exact HMM"):
        inference.stop_disabled_inference(config)

    train_source = TRAIN_SOURCE.read_text()
    inference_source = INFERENCE_SOURCE.read_text()
    for heading in [
        "## 2. Notebook-safe runtime, configuration, path, and SHA helpers",
        "## 3. Frozen scientific and execution contract",
        "## 4. Truth-free exp209 input and signed-path helpers",
        "## 5. Sixteen-well exact-state resource projection",
        "## 6. Candidate generation and pre-truth SHA freeze",
        "## 7. Late truth and hidden-like attachment",
        "## 8. Stage 0 metrics and promotion gates",
        "## 9. Execution orchestration and generated artifacts",
        "## 10. Setup and fail-closed execution selection",
    ]:
        assert heading in train_source
    assert "from settings import" not in train_source
    assert "__file__" not in train_source
    assert "__file__" not in inference_source
