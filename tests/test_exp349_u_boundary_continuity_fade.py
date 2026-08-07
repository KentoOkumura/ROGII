from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = (
    ROOT / "experiments" / "exp349_exp287_u_boundary_continuity_fade"
)
TRAIN_SOURCE = (
    EXPERIMENT_DIR
    / "exp349_exp287_u_boundary_continuity_fade_compact_selfcontained_train.py"
)
INFERENCE_SOURCE = (
    EXPERIMENT_DIR
    / "exp349_exp287_u_boundary_continuity_fade_compact_selfcontained_inference.py"
)


def load_source(path: Path, module_name: str):
    os.environ["EXP349_IMPORT_ONLY"] = "1"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


exp349 = load_source(TRAIN_SOURCE, "exp349_train_test_module")
exp349_inference = load_source(INFERENCE_SOURCE, "exp349_inference_test_module")


def test_config_preserves_one_postprocess_and_zero_training_contract() -> None:
    contract = exp349.validate_scientific_contract(exp349.CONFIG, require_execution=False)
    assert contract == {
        "stage": exp349.CONFIG["execution"]["active_stage"],
        "postprocess_variants": 1,
        "reporting_folds": 5,
        "trained_folds": 0,
        "model_configs": 0,
        "trained_models": 0,
        "boosters": 0,
        "pf_well_runs": 0,
        "beam_well_runs": 0,
        "hmm_well_runs": 0,
        "parent_control_retraining": False,
        "gpu": False,
    }
    execution = exp349.CONFIG["execution"]
    assert contract["stage"] in execution["allowed_stages"]
    if contract["stage"] == "implementation_complete_no_run":
        assert execution["run_approved"] is False
        assert execution["kaggle_push_approved"] is False
        assert execution["run_stage_a_generation"] is False
        assert execution["run_stage_b_evaluation"] is False
    else:
        assert contract["stage"] == "stage0_saved_oof_audit"
        assert execution["run_approved"] is True
        assert execution["kaggle_push_approved"] is True
        assert execution["run_stage_a_generation"] is True
        assert execution["run_stage_b_evaluation"] is True
    assert exp349.CONFIG["implementation"]["approvals"]["implementation"] is True
    assert exp349.CONFIG["implementation"]["approvals"]["canonical_notebook_adoption"] is True
    assert exp349.CONFIG["implementation"]["approvals"]["kaggle_stage0_run"] is True


def test_inference_contract_is_fail_closed() -> None:
    contract = exp349_inference.validate_zero_inference_contract(exp349_inference.CONFIG)
    assert contract["postprocess_variants"] == 1
    assert contract["boosters"] == 0
    assert contract["run_inference"] is False
    assert contract["create_submission"] is False
    assert contract["submit_to_kaggle"] is False


def test_sha_resolver_accepts_equivalent_copies_in_pattern_order(tmp_path: Path) -> None:
    first = tmp_path / "first.parquet"
    second = tmp_path / "second.parquet"
    different = tmp_path / "different.parquet"
    first.write_bytes(b"same")
    second.write_bytes(b"same")
    different.write_bytes(b"different")
    expected = exp349.sha256_file(first)
    selected = exp349.select_first_sha_matched_file(
        [first, second, different],
        expected_sha256=expected,
        label="synthetic equivalent input",
    )
    assert selected == first.resolve()
    with pytest.raises(FileNotFoundError, match="no SHA-matched"):
        exp349.select_first_sha_matched_file(
            [different],
            expected_sha256=expected,
            label="synthetic missing input",
        )


def raw_well_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "MD": [0.0, 10.0, 20.0, 30.0, 40.0],
            "Z": [100.0, 99.0, 98.0, 97.0, 96.0],
            "TVT_input": [10.0, 11.0, np.nan, np.nan, np.nan],
        }
    )


def test_raw_suffix_contract_and_fixed_formula() -> None:
    raw, raw_manifest = exp349.build_raw_suffix_for_well("well_a", raw_well_frame())
    assert raw["id"].tolist() == ["well_a_2", "well_a_3", "well_a_4"]
    assert raw_manifest["prefix_rows"] == 2
    assert raw_manifest["suffix_rows"] == 3
    assert raw["u_last"].eq(110.0).all()
    prediction_column = exp349.CONFIG["validation"]["parent_prediction_column"]
    parent = pd.DataFrame(
        {
            "id": raw["id"],
            "well": "well_a",
            prediction_column: [20.0, 21.0, 22.0],
        }
    )
    candidate, diagnostic, technical = exp349.apply_fixed_u_boundary_fade(
        raw,
        parent,
        exp349.CONFIG,
    )
    gap = 20.0 + 98.0 - 110.0
    expected_move = -8.0 * np.exp(-raw["md_since_boundary"].to_numpy() / 240.0)
    assert candidate["gap_u"].to_numpy() == pytest.approx(np.full(3, gap))
    assert candidate["move_tvt"].to_numpy() == pytest.approx(expected_move)
    assert candidate["candidate_pred_tvt"].to_numpy() == pytest.approx(
        parent[prediction_column].to_numpy() + expected_move
    )
    assert diagnostic.loc[0, "abs_gap_bucket"] == "[8,+inf)"
    assert diagnostic.loc[0, "gap_sign"] == "positive"
    assert diagnostic.loc[0, "gap_u_after_first_hidden"] == pytest.approx(
        gap + expected_move[0]
    )
    assert technical["maximum_abs_move_tvt"] < 8.0
    assert technical["abs_move_nonincreasing_per_well"] is True


@pytest.mark.parametrize(
    "frame,error",
    [
        (
            pd.DataFrame(
                {
                    "MD": [0.0, 1.0, 2.0],
                    "Z": [0.0, 0.0, 0.0],
                    "TVT_input": [1.0, np.nan, 2.0],
                }
            ),
            "finite prefix plus contiguous NaN suffix",
        ),
        (
            pd.DataFrame(
                {
                    "MD": [0.0, 2.0, 1.0],
                    "Z": [0.0, 0.0, 0.0],
                    "TVT_input": [1.0, 2.0, np.nan],
                }
            ),
            "strictly increasing",
        ),
    ],
)
def test_raw_suffix_rejects_contract_violations(frame: pd.DataFrame, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        exp349.build_raw_suffix_for_well("bad", frame)


def test_pretruth_schema_rejects_truth_fold_and_assignment() -> None:
    for forbidden in [
        "actual_tvt",
        "outer_fold",
        "error",
        "verification_like_spatial_role",
    ]:
        with pytest.raises(ValueError, match="pretruth-forbidden"):
            exp349.validate_pretruth_columns(["id", "well", forbidden], label="synthetic")
    exp349.validate_pretruth_columns(
        ["id", "well", "parent_pred_tvt", "candidate_pred_tvt"],
        label="synthetic allowed",
    )


def test_raw_parent_identity_mismatch_fails_closed() -> None:
    raw, _ = exp349.build_raw_suffix_for_well("well_a", raw_well_frame())
    prediction_column = exp349.CONFIG["validation"]["parent_prediction_column"]
    parent = pd.DataFrame(
        {
            "id": ["wrong_0", "wrong_1", "wrong_2"],
            "well": "well_a",
            prediction_column: [20.0, 21.0, 22.0],
        }
    )
    with pytest.raises(ValueError, match="unmatched or nonfinite"):
        exp349.apply_fixed_u_boundary_fade(raw, parent, exp349.CONFIG)


def synthetic_evaluation_frame(*, worsen_far: bool = False) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for fold in range(5):
        well = f"well_{fold}"
        for row, distance in enumerate([100.0, 300.0, 700.0, 1100.0]):
            candidate = 9.0 if distance < 240.0 else 10.0
            if worsen_far and distance >= 1000.0:
                candidate = 10.1
            rows.append(
                {
                    "id": f"{well}_{row}",
                    "well": well,
                    "outer_fold": fold,
                    "actual_tvt": 0.0,
                    "parent_pred_tvt": 10.0,
                    "candidate_pred_tvt": candidate,
                    "md_since_boundary": distance,
                    "gap_u": 5.0,
                    "abs_gap_bucket": "[4,8)",
                    "gap_sign": "positive",
                    "verification_like_spatial_role": "valid",
                    "verification_like_typewell_purged_role": "valid",
                }
            )
    return pd.DataFrame(rows)


def test_fixed_scientific_gate_passes_only_all_registered_checks() -> None:
    tables, by_well, scientific = exp349.evaluate_fixed_candidate(
        synthetic_evaluation_frame(),
        exp349.CONFIG,
    )
    assert set(tables) == {
        "pooled_metrics.csv",
        "fold_metrics.csv",
        "distance_bucket_metrics.csv",
        "gap_bucket_metrics.csv",
        "gap_sign_metrics.csv",
        "hidden_like_metrics.csv",
    }
    assert scientific["pooled_gain_parent_minus_candidate"] > 0.02
    assert scientific["improving_folds"] == 5
    assert scientific["all_scientific_checks_passed"] is True
    assert len(by_well) == 5

    _, _, far_regression = exp349.evaluate_fixed_candidate(
        synthetic_evaluation_frame(worsen_far=True),
        exp349.CONFIG,
    )
    assert far_regression["checks"]["maximum_1000_plus_rmse_delta_ft"] is False
    assert far_regression["all_scientific_checks_passed"] is False


def test_freeze_barrier_detects_artifact_tampering(tmp_path: Path) -> None:
    raw, raw_manifest_row = exp349.build_raw_suffix_for_well("well_a", raw_well_frame())
    prediction_column = exp349.CONFIG["validation"]["parent_prediction_column"]
    parent = pd.DataFrame(
        {
            "id": raw["id"],
            "well": "well_a",
            prediction_column: [20.0, 21.0, 22.0],
        }
    )
    candidate, diagnostics, technical = exp349.apply_fixed_u_boundary_fade(
        raw,
        parent,
        exp349.CONFIG,
    )
    raw_manifest = pd.DataFrame(
        [
            {
                **raw_manifest_row,
                "source_file": "well_a.csv",
                "source_file_sha256": "a" * 64,
            }
        ]
    )
    parent_path = tmp_path / "parent.parquet"
    parent_path.write_bytes(b"synthetic parent")
    model_path = tmp_path / "model_manifest.json"
    model_path.write_text(json.dumps({"model_count": 15}))
    config = copy.deepcopy(exp349.CONFIG)
    config["validation"]["parent_model_manifest_sha256"] = exp349.sha256_file(model_path)
    exp349.write_json(tmp_path / "scientific_contract.json", {"synthetic": True})
    _, freeze_sha = exp349.freeze_target_free_generation(
        candidate,
        diagnostics,
        raw_manifest,
        config=config,
        config_path=exp349.CONFIG_PATH,
        parent_oof_path=parent_path,
        parent_model_manifest_path=model_path,
        parent_audit={
            "opened_columns": exp349.PRETRUTH_PARENT_COLUMNS,
            "truth_columns_opened": [],
            "outer_fold_opened": False,
        },
        raw_audit={
            "opened_columns": exp349.RAW_ALLOWED_COLUMNS,
            "truth_columns_opened": [],
        },
        technical_audit=technical,
        output_dir=tmp_path,
    )
    freeze = exp349.verify_generation_freeze(
        tmp_path,
        expected_freeze_manifest_sha256=freeze_sha,
    )
    assert freeze["truth_access_before_freeze_count"] == 0
    diagnostic_path = tmp_path / "u_boundary_diagnostics_pretruth.csv"
    diagnostic_path.write_text(diagnostic_path.read_text().replace("well_a", "tampered", 1))
    with pytest.raises(ValueError, match="SHA mismatch"):
        exp349.verify_generation_freeze(
            tmp_path,
            expected_freeze_manifest_sha256=freeze_sha,
        )
