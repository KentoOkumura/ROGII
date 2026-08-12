from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    ROOT
    / "experiments"
    / "exp367_stratified_signed_curvature_pf"
    / "exp367_stratified_signed_curvature_pf_compact_selfcontained_train.py"
)
INFERENCE_MODULE_PATH = (
    ROOT
    / "experiments"
    / "exp367_stratified_signed_curvature_pf"
    / "exp367_stratified_signed_curvature_pf_compact_selfcontained_inference.py"
)
SPEC = importlib.util.spec_from_file_location("exp367_stage0", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
exp367 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = exp367
SPEC.loader.exec_module(exp367)
INFERENCE_SPEC = importlib.util.spec_from_file_location(
    "exp367_inference",
    INFERENCE_MODULE_PATH,
)
assert INFERENCE_SPEC is not None and INFERENCE_SPEC.loader is not None
exp367_inference = importlib.util.module_from_spec(INFERENCE_SPEC)
sys.modules[INFERENCE_SPEC.name] = exp367_inference
INFERENCE_SPEC.loader.exec_module(exp367_inference)


def test_frozen_scientific_contract_matches_config() -> None:
    package_dir = MODULE_PATH.parent
    config = exp367.load_config(package_dir)
    contract = exp367.validate_scientific_contract(config)

    assert contract["candidate_signs"] == [-1, 0, 1]
    assert contract["block_rows"] == 512
    assert contract["pf_seed_well_runs"] == 0
    assert len(contract["contract_sha256"]) == 64


def test_inference_contract_is_fail_closed() -> None:
    config = exp367_inference.load_config(INFERENCE_MODULE_PATH.parent)
    status = exp367_inference.validate_inference_is_disabled(config)

    assert status["implementation_scope"] == "stage0_only"
    assert status["run_stage_1"] is False
    assert status["run_inference"] is False
    assert status["create_submission"] is False


def test_stable_seed_is_keyed_and_reproducible() -> None:
    first = exp367.stable_seed(exp367.EXPERIMENT_NAME, "well-a", "pf", 0)
    repeated = exp367.stable_seed(exp367.EXPERIMENT_NAME, "well-a", "pf", 0)
    next_seed = exp367.stable_seed(exp367.EXPERIMENT_NAME, "well-a", "pf", 1)

    assert first == repeated
    assert first != next_seed
    assert 1 <= first <= 2_147_483_647


def test_fixed_signed_paths_preserve_expected_sign_order() -> None:
    rows = 600
    md = np.arange(1.0, rows + 1.0)
    z = np.zeros(rows)
    paths = exp367.generate_fixed_signed_paths(
        eval_md=md,
        eval_z=z,
        last_known_md=0.0,
        last_known_position=1000.0,
        initial_rate=0.0,
        signs=[-1, 0, 1],
        momentum=0.998,
        rate_drift_per_row=0.000009765625,
        typewell_min=0.0,
        typewell_max=2000.0,
        support_pad=100.0,
    )

    assert paths[-1][-1] < paths[0][-1] < paths[1][-1]
    assert np.allclose(paths[0], 1000.0)
    assert np.isfinite(np.stack(list(paths.values()))).all()


def test_block_and_circular_control_contract() -> None:
    blocks = exp367.fixed_full_blocks(1024, block_rows=512, stride_rows=256)
    assert blocks == [(0, 512), (256, 768), (512, 1024)]

    values = [np.full(512, index, dtype=float) for index in range(3)]
    shifted = exp367.circular_control_gr_blocks(
        values,
        shift_blocks=1,
        single_block_shift_rows=256,
    )
    assert [float(value[0]) for value in shifted] == [1.0, 2.0, 0.0]


def test_score_ties_use_zero_first_order() -> None:
    ranking = exp367.ranked_signs(
        {-1: -1.0, 0: -1.0, 1: -1.0},
        tie_break=[0, -1, 1],
    )
    assert ranking == [0, -1, 1]


def test_truth_access_ledger_rejects_prefreeze_target() -> None:
    ledger = exp367.TruthAccessLedger()
    with pytest.raises(ValueError, match="forbidden pre-freeze"):
        ledger.guard_prefreeze_columns(["MD", "TVT"], 12, "synthetic")
    assert ledger.truth_rows_before_freeze == 12


def test_prefreeze_well_builder_never_attaches_horizontal_truth(
    tmp_path: Path,
) -> None:
    well = "synthetic"
    prefix_rows = 40
    suffix_rows = 520
    total_rows = prefix_rows + suffix_rows
    md = np.arange(total_rows, dtype=float)
    z = -0.25 * md
    true_tvt = 1000.0 + 0.25 * md
    horizontal = pd.DataFrame(
        {
            "MD": md,
            "Z": z,
            "GR": 80.0 + 10.0 * np.sin(md / 30.0),
            "TVT_input": np.where(md < prefix_rows, true_tvt, np.nan),
            "TVT": true_tvt,
        }
    )
    typewell_tvt = np.linspace(900.0, 1200.0, 601)
    typewell = pd.DataFrame(
        {
            "TVT": typewell_tvt,
            "GR": 80.0 + 10.0 * np.sin((typewell_tvt - 1000.0) / 30.0),
        }
    )
    horizontal.to_csv(tmp_path / f"{well}__horizontal_well.csv", index=False)
    typewell.to_csv(tmp_path / f"{well}__typewell.csv", index=False)
    config = exp367.load_config(MODULE_PATH.parent)
    ledger = exp367.TruthAccessLedger()

    candidate, scores, manifest = exp367.build_prefreeze_rows_for_well(
        well,
        tmp_path,
        config,
        ledger,
    )

    assert len(candidate) == suffix_rows
    assert len(scores) == 1
    assert "TVT" not in candidate.columns
    assert "true_tvt" not in candidate.columns
    assert manifest["status"] == "ok"
    assert ledger.truth_rows_before_freeze == 0


def test_ranking_metrics_reports_positive_zero_path_gain() -> None:
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
    metrics = exp367.ranking_metrics(frame)

    assert metrics["top1_gain_vs_zero_first"] == pytest.approx(0.5)
    assert metrics["mrr_gain_vs_zero_first"] == pytest.approx(0.25)
    assert metrics["selected_path_rmse_gain_vs_zero_ft"] > 0.0
