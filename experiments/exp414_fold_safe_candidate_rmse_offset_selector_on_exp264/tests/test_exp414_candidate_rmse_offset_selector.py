from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from src.candidate_rmse_offset_selector import (
    build_candidate_rmse_offsets,
    offsets_for_labels,
    reconstruct_abs_error_score,
    validate_candidate_rmse_offset_config,
    validate_static_contract,
    write_candidate_rmse_offset_artifacts,
)
from src.candidate_rmse_root_cause_readout import run_root_cause_readout


ROOT = Path(__file__).resolve().parents[3]
EXP = "exp414_fold_safe_candidate_rmse_offset_selector_on_exp264"
EXP_DIR = ROOT / "experiments" / EXP
CONTRACT_PATH = (
    ROOT
    / "experiments"
    / "exp407_fold_safe_inverse_rmse_weighted_dual_selector_on_exp264"
    / "candidate_contract.yaml"
)


def _labels(candidate_order: list[str], partition: int = 0) -> pd.DataFrame:
    errors = np.asarray(
        [
            [1.0, 3.0, 5.0],
            [2.0, 4.0, 6.0],
            [3.0, 5.0, 7.0],
        ],
        dtype=np.float64,
    )
    rows = []
    for base_row in range(len(errors)):
        for position, candidate_id in enumerate(candidate_order):
            rows.append(
                {
                    "id": f"id_{partition}_{base_row}",
                    "well": f"well_{partition}",
                    "well_row_idx": base_row,
                    "outer_fold": partition,
                    "candidate_id": candidate_id,
                    "candidate_abs_error": errors[base_row, position],
                }
            )
    return pd.DataFrame(rows)


def test_candidate_rmse_offset_is_additive_and_exact() -> None:
    candidates = ["a", "b", "c"]
    labels = _labels(candidates)
    result = build_candidate_rmse_offsets(
        labels,
        candidates,
        partition_id=4,
    )
    expected = np.sqrt(
        np.mean(
            np.square(
                labels["candidate_abs_error"].to_numpy().reshape(-1, 3)
            ),
            axis=0,
        )
    )
    assert np.allclose(result.offset, expected)
    assert np.allclose(
        result.residual_target + np.tile(result.offset, 3),
        labels["candidate_abs_error"].to_numpy(),
    )
    assert result.audit["training_sample_weight_applied"] is False
    assert result.audit["binary_model_fit_count"] == 0
    assert result.table["offset_scale"].eq(1.0).all()


def test_candidate_rmse_offset_rejects_candidate_order_drift() -> None:
    candidates = ["a", "b", "c"]
    labels = _labels(candidates)
    labels.loc[[0, 1], "candidate_id"] = ["b", "a"]
    with pytest.raises(ValueError, match="candidate-long order"):
        build_candidate_rmse_offsets(labels, candidates, partition_id=0)


def test_offsets_for_valid_labels_and_reconstruction_clip() -> None:
    candidates = ["a", "b", "c"]
    labels = _labels(candidates)
    offset = np.asarray([1.0, 2.0, 3.0])
    row_offset = offsets_for_labels(labels, candidates, offset)
    assert np.array_equal(row_offset, np.tile(offset, 3))
    score = reconstruct_abs_error_score(
        np.asarray([-2.0, -1.0, 1.0]),
        np.asarray([1.0, 2.0, 3.0]),
    )
    assert np.array_equal(score, np.asarray([0.0, 1.0, 4.0]))


def test_offset_manifest_records_no_weight_or_forbidden_truth(tmp_path: Path) -> None:
    candidates = ["a", "b", "c"]
    results = []
    for fold in range(5):
        result = build_candidate_rmse_offsets(
            _labels(candidates, partition=fold),
            candidates,
            partition_id=fold,
        )
        result.audit["fit_valid_well_overlap"] = 0
        results.append(result)
    manifest = write_candidate_rmse_offset_artifacts(tmp_path, results)
    assert manifest["all_checks_passed"] is True
    assert manifest["training_sample_weight_applied"] is False
    assert manifest["binary_model_fit_count"] == 0
    assert manifest["truth_read_ledger"]["forbidden_truth_reads"] == 0
    assert (tmp_path / "candidate_rmse_offset_by_fold.csv").exists()


def test_exp414_static_contract_is_frozen() -> None:
    config = yaml.safe_load((EXP_DIR / "config.yaml").read_text())
    contract = yaml.safe_load(CONTRACT_PATH.read_text())
    static = validate_static_contract(config, contract)
    assert static["candidate_count"] == 12
    assert static["feature_count"] == 88
    assert static["cost"]["planned_cpu_boosters"] == 5
    assert static["cost"]["classifier_boosters"] == 0
    validate_candidate_rmse_offset_config(config["candidate_rmse_offset"])


def _surface_frame(
    *,
    candidate_order: list[str],
    fold: int,
    parent: bool,
    treatment: bool = False,
) -> pd.DataFrame:
    rows = []
    truth_errors = np.asarray(
        [
            [1.0, 4.0, 2.0],
            [3.0, 1.0, 2.0],
            [2.0, 5.0, 1.0],
            [4.0, 2.0, 1.0],
        ]
    )
    parent_score = np.asarray(
        [
            [1.0, 3.0, 2.0],
            [2.0, 1.0, 3.0],
            [2.0, 3.0, 1.0],
            [3.0, 2.0, 1.0],
        ]
    )
    if parent:
        score = parent_score
    elif treatment:
        score = parent_score + np.asarray([0.05, -0.03, 0.01])[None, :]
    else:
        row_noise = np.asarray(
            [
                [0.8, -0.8, 0.1],
                [-0.6, 0.7, 0.0],
                [0.7, -0.7, 0.2],
                [-0.8, 0.8, -0.1],
            ]
        )
        score = parent_score + row_noise
    for base_row in range(4):
        for position, candidate_id in enumerate(candidate_order):
            row = {
                "id": f"id_{fold}_{base_row}",
                "well": f"well_{fold}",
                "well_row_idx": base_row,
                "outer_fold": fold,
                "md_since": float(100 + base_row * 500),
                "candidate_id": candidate_id,
                "candidate_tvt": 1000.0 + position,
                "actual_abs_error": truth_errors[base_row, position],
                "pred_abs_error": score[base_row, position],
            }
            if not treatment:
                row["actual_within10"] = 1
                row["p_within10"] = 0.8 if parent else 0.75
            rows.append(row)
    return pd.DataFrame(rows)


def _write_row_groups(path: Path, frames: list[pd.DataFrame]) -> None:
    writer = None
    try:
        for frame in frames:
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(path, table.schema)
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()


def test_root_cause_readout_runs_with_aligned_synthetic_surfaces(
    tmp_path: Path,
) -> None:
    candidates = ["a", "b", "c"]
    parent_path = tmp_path / "parent.parquet"
    exp407_path = tmp_path / "exp407.parquet"
    treatment_path = tmp_path / "treatment.parquet"
    _write_row_groups(
        parent_path,
        [
            _surface_frame(
                candidate_order=candidates, fold=fold, parent=True
            )
            for fold in range(2)
        ],
    )
    _write_row_groups(
        exp407_path,
        [
            _surface_frame(
                candidate_order=candidates, fold=fold, parent=False
            )
            for fold in range(2)
        ],
    )
    _write_row_groups(
        treatment_path,
        [
            _surface_frame(
                candidate_order=candidates,
                fold=fold,
                parent=False,
                treatment=True,
            )
            for fold in range(2)
        ],
    )
    weight_rows = []
    for fold in range(2):
        for position, candidate_id in enumerate(candidates):
            weight_rows.append(
                {
                    "fit_partition": fold,
                    "candidate_position": position,
                    "candidate_id": candidate_id,
                    "fit_candidate_rmse": 1.0 + position,
                    "final_weight": 1.2 - position * 0.2,
                }
            )
    weight_path = tmp_path / "weights.csv"
    pd.DataFrame(weight_rows).to_csv(weight_path, index=False)
    config = {
        "validation": {"outer_folds": 2},
        "root_cause_gate": {
            "global_shift_only_hard_rmse_max": 100.0,
            "global_shift_only_nonworse_folds_min": 0,
            "local_change_only_hard_rmse_min": 0.0,
            "local_change_only_nonworse_folds_max": 2,
            "final_weight_delta_pred_std_spearman_max": 1.0,
            "final_weight_delta_score_mae_spearman_max": 1.0,
            "final_weight_delta_logloss_spearman_max": 1.0,
            "final_weight_delta_pred_mean_spearman_abs_max": 1.0,
            "confident_margin_net_damage_share_min": -100.0,
        },
    }
    summary = run_root_cause_readout(
        parent_path=parent_path,
        exp407_path=exp407_path,
        treatment_path=treatment_path,
        exp407_weight_table_path=weight_path,
        candidate_order=candidates,
        primary_domain=candidates[:2],
        output_dir=tmp_path / "output",
        config=config,
    )
    assert summary["switch_summary"]["base_rows"] == 8
    assert len(summary["selector_counterfactual_pooled"]) == 5
    assert (
        summary["treatment_instability"]["treatment_mean_centered_delta_std"]
        < 1.0e-12
    )
    written = json.loads(
        (tmp_path / "output" / "root_cause_summary.json").read_text()
    )
    assert written["candidate_order"] == candidates
