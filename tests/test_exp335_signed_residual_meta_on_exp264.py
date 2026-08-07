from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.candidate_selector_pipeline import build_compact_meta, candidate_ids
from src.signed_residual_meta import (
    ParquetBatchCursor,
    add_signed_residual_labels,
    build_signed_compact_meta,
    evaluate_signed_residual_gate,
    parent_compact_columns,
    signed_compact_feature_names,
    signed_compact_schema,
    stage_d_cost_contract,
    stage_d_retained_base_columns,
    stage_s_cost_contract,
)

ROOT = Path(__file__).resolve().parents[1]
EXP264 = ROOT / "experiments" / "exp264_exp263_candidate_confidence_dual_selector"
EXP335 = ROOT / "experiments" / "exp335_signed_residual_meta_on_exp264"


def load_contract() -> dict:
    return yaml.safe_load((EXP264 / "candidate_contract.yaml").read_text())


def load_config() -> dict:
    return yaml.safe_load((EXP335 / "config.yaml").read_text())


def synthetic_surfaces() -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    contract = load_contract()
    ids = candidate_ids(contract)
    rows = 5
    base = pd.DataFrame(
        {
            "id": [f"row_{index}" for index in range(rows)],
            "well": ["well_a"] * rows,
            "well_row_idx": np.arange(10, 10 + rows, dtype=np.int32),
            "outer_fold": np.zeros(rows, dtype=np.int8),
            "md_since": np.arange(1, rows + 1, dtype=np.float32),
            "last_known_tvt": np.full(rows, 1000.0, dtype=np.float32),
        }
    )
    values = (
        base["last_known_tvt"].to_numpy(np.float32)[:, None]
        + np.arange(1, len(ids) + 1, dtype=np.float32)[None, :]
        + np.arange(rows, dtype=np.float32)[:, None] / 10.0
    )
    pred_abs_error = np.tile(
        np.asarray([8, 1, 4, 2, 7, 6, 5, 3, 9, 10, 11, 0], dtype=np.float32),
        (rows, 1),
    )
    p_within10 = np.tile(
        np.asarray(
            [0.1, 0.7, 0.4, 0.8, 0.2, 0.3, 0.5, 0.6, 0.15, 0.25, 0.35, 0.9],
            dtype=np.float32,
        ),
        (rows, 1),
    )
    saved = build_compact_meta(
        base,
        values,
        pred_abs_error,
        p_within10,
        np.ones_like(values, dtype=bool),
        np.ones_like(values, dtype=bool),
        contract,
    )
    pred_signed = np.tile(
        np.arange(-6, 6, dtype=np.float32)[None, :], (rows, 1)
    )
    return base, values, saved, pred_signed


def test_stage_d_retained_base_columns_deduplicates_anchor_feature() -> None:
    columns = stage_d_retained_base_columns(
        ["last_known_tvt", "pf_ancc", "md_since", "another_feature"]
    )
    assert columns == [
        "id",
        "well",
        "target",
        "last_known_tvt",
        "md_since",
        "pf_ancc",
        "another_feature",
    ]
    assert len(columns) == len(set(columns))


def test_signed_label_formula_is_explicit_and_fail_closed() -> None:
    contract = load_contract()
    base, values, _, _ = synthetic_surfaces()
    ids = candidate_ids(contract)
    metadata = pd.DataFrame(
        {
            "candidate_tvt": values.reshape(-1),
            "candidate_available": True,
            "candidate_id": np.tile(ids, len(base)),
        }
    )
    truth = np.arange(len(base), dtype=np.float32) + np.float32(1015.0)
    labels, evidence = add_signed_residual_labels(metadata, truth, len(ids))
    expected = np.repeat(truth, len(ids)) - values.reshape(-1)
    np.testing.assert_allclose(labels["candidate_signed_residual"], expected)
    assert evidence["formula"] == "true_tvt-candidate_tvt"
    assert evidence["formula_parity_passed"] is True

    unavailable = metadata.copy()
    unavailable.loc[0, "candidate_available"] = False
    with pytest.raises(ValueError, match="implicit zero label"):
        add_signed_residual_labels(unavailable, truth, len(ids))


def test_signed_compact_is_exactly_23_add_only_features() -> None:
    contract = load_contract()
    base, values, saved, pred_signed = synthetic_surfaces()
    compact, evidence = build_signed_compact_meta(
        base,
        values,
        pred_signed,
        saved[parent_compact_columns(contract)],
        contract,
    )
    selector_columns = [column for column in compact if column.startswith("selector__")]
    assert selector_columns == signed_compact_feature_names(contract)
    assert len(selector_columns) == 23
    assert evidence["passed"] is True
    assert not any("corrected_tvt__" in column for column in selector_columns)
    assert not any("hard_top1" in column for column in selector_columns)
    schema = signed_compact_schema(contract)
    assert schema["feature_count"] == 23
    assert schema["features"] == selector_columns

    ids = candidate_ids(contract)
    primary = contract["legal_domains"]["primitive_pair_bank"]["candidates"]
    primary_positions = [ids.index(candidate_id) for candidate_id in primary]
    abs_scores = saved[
        [f"selector__pred_abs_error__{candidate_id}" for candidate_id in ids]
    ].to_numpy()
    selected = np.asarray(primary_positions)[np.argmin(abs_scores[:, primary_positions], axis=1)]
    expected_signed = pred_signed[np.arange(len(base)), selected]
    np.testing.assert_array_equal(
        compact[
            "selector__primitive_pair_bank__pred_abs_error__signed_residual_at_top1"
        ],
        expected_signed,
    )


def test_signed_compact_rejects_parent_key_or_top1_drift() -> None:
    contract = load_contract()
    base, values, saved, pred_signed = synthetic_surfaces()
    parent = saved[parent_compact_columns(contract)].copy()
    parent.loc[0, "id"] = "wrong"
    with pytest.raises(ValueError, match="key mismatch"):
        build_signed_compact_meta(base, values, pred_signed, parent, contract)

    parent = saved[parent_compact_columns(contract)].copy()
    column = "selector__primitive_pair_bank__pred_abs_error__top1_value"
    parent.loc[0, column] += np.float32(1.0)
    with pytest.raises(AssertionError, match="top1 identity parity"):
        build_signed_compact_meta(base, values, pred_signed, parent, contract)


def test_stage_s_gate_requires_pooled_and_four_of_five_folds() -> None:
    passing = pd.DataFrame(
        {
            "fold": np.arange(5),
            "long_rows": np.full(5, 100),
            "signed_sse": [81, 81, 81, 81, 121],
            "prior_signed_sse": [100, 100, 100, 100, 100],
        }
    )
    gate = evaluate_signed_residual_gate(passing, minimum_improved_outer_folds=4)
    assert gate["pooled_improved"] is True
    assert gate["improved_outer_folds"] == 4
    assert gate["passed"] is True

    only_three = passing.copy()
    only_three.loc[3, "signed_sse"] = 101
    assert evaluate_signed_residual_gate(
        only_three, minimum_improved_outer_folds=4
    )["passed"] is False


def test_parquet_cursor_preserves_exact_chunk_alignment(tmp_path: Path) -> None:
    frame = pd.DataFrame({"id": [f"row_{index}" for index in range(11)], "value": range(11)})
    path = tmp_path / "parent.parquet"
    frame.to_parquet(path, index=False)
    cursor = ParquetBatchCursor(path, ["id", "value"], batch_size=4)
    observed = pd.concat([cursor.take(3), cursor.take(5), cursor.take(3)], ignore_index=True)
    cursor.finish()
    pd.testing.assert_frame_equal(observed, frame)


def test_config_records_stage_s_and_stage_d_complete_before_inference() -> None:
    config = load_config()
    cost = stage_s_cost_contract(config)
    assert cost["planned_cpu_boosters"] == 20
    assert cost["existing_selector_retraining_boosters"] == 0
    assert cost["downstream_gpu_boosters"] == 0
    assert config["execution"]["implementation_complete"] is True
    assert config["execution"]["kaggle_package_approved"] is True
    assert config["execution"]["kaggle_push_approved"] is True
    assert config["execution"]["preflight_run_approved"] is True
    assert config["execution"]["stage"] == "cpu_inference"
    assert config["execution"]["selector_train_approved"] is True
    assert config["execution"]["selector_train_completed"] is True
    assert config["execution"]["run_selector_train"] is False
    assert config["execution"]["downstream_train_approved"] is True
    assert config["execution"]["run_downstream_train"] is False
    assert (
        config["execution"]["downstream_train_status"]
        == "version2_complete_guard_failed_closed"
    )
    assert config["execution"]["inference_approved"] is True
    assert config["execution"]["submission_approved"] is False

    source = (
        EXP335
        / "exp335_signed_residual_meta_on_exp264_compact_selfcontained_train.py"
    ).read_text()
    assert "preflight_run_approved" in source
    assert "selector_train_approved" in source
    assert "run_stage_s_preflight" in source
    assert "run_stage_s(" in source
    assert 'KAGGLE_INPUT_ROOT / "competitions" / slug' in source
    assert "__file__" not in source
    assert source.count("# %% [markdown]") >= 8

    stage_d = stage_d_cost_contract(config)
    assert stage_d["active_variants"] == ["signed_residual_meta_addonly"]
    assert stage_d["lightgbm_config_indices"] == [0, 1, 2]
    assert stage_d["folds"] == 5
    assert stage_d["planned_gpu_boosters"] == 15
    assert config["model"]["downstream_tvt"]["execution_count"]["actual_gpu_boosters"] == 15
    assert stage_d["saved_control_retraining_boosters"] == 0
    assert stage_d["feature_counts"] == {
        "clean_base": 273,
        "saved_exp264_compact": 74,
        "signed_residual_compact": 23,
        "final": 370,
    }

    stage_d_source = (
        EXP335 / "exp335_signed_residual_meta_on_exp264_tvt_train.py"
    ).read_text()
    assert "run_stage_d(" in stage_d_source
    assert "planned_gpu_boosters" in stage_d_source
    assert "saved_exp264_control_retraining" in stage_d_source
    assert "Inference executed: False" in stage_d_source
    assert "Submission generated or submitted: False" in stage_d_source
    assert "__file__" not in stage_d_source
    assert stage_d_source.count("# %% [markdown]") >= 8


def test_cpu_inference_override_keeps_guard_failure_and_submit_disabled() -> None:
    config = load_config()
    execution = config["execution"]
    inference = config["inference"]

    assert execution["stage"] == "cpu_inference"
    assert execution["inference_approved"] is True
    # The package embeds these as true, but the root config is disarmed after push.
    assert execution["run_inference"] is False
    assert execution["create_submission"] is False
    assert execution["submit_to_kaggle"] is False
    assert inference["status"] == "user_authorized_2026_07_23_cpu"
    assert inference["runtime"] == "kaggle_cpu"
    assert inference["booster_training_count"] == 0
    assert inference["stage_d_scientific_support_passed"] is False
    assert inference["stage_d_promotion_passed"] is False
    assert inference["competition_submit_authorized"] is False
    assert inference["parent_selector_model_count"] == 40
    assert inference["signed_selector_model_count"] == 20
    assert inference["tvt_model_count"] == 15
    assert inference["expected_base_feature_count"] == 273
    assert inference["expected_parent_compact_feature_count"] == 74
    assert inference["expected_signed_compact_feature_count"] == 23
    assert inference["expected_final_feature_count"] == 370
    assert config["runtime"]["kaggle"]["inference"]["enable_gpu"] is False

    source = (
        EXP335
        / "exp335_signed_residual_meta_on_exp264_compact_selfcontained_inference.py"
    ).read_text()
    assert "build_signed_compact_meta(" in source
    assert "signed_selector_models[outer]" in source
    assert "parent_compact_features + signed_compact_features" in source
    assert '"runtime": "kaggle_cpu"' in source
    assert '"competition_submit_authorized": False' in source
    assert "kaggle competitions submit" not in source
    assert "__file__" not in source
    assert source.count("# %% [markdown]") >= 8
