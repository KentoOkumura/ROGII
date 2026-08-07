from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.candidate_selector_pipeline import FoldBundle, contract_by_id
from src.pf_ancc_selector_audit import (
    augment_fold_bundle,
    build_base_contract,
    build_variant_contract,
    downstream_guard,
    raw_test_only_schema_guard,
    schema_sha,
)

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments/exp277_pf_ancc_small_seed_mean_addonly_selector_audit"


def full_contract() -> dict:
    return yaml.safe_load((EXP / "candidate_contract.yaml").read_text())


def synthetic_base_bundle(contract: dict) -> FoldBundle:
    base_contract = build_base_contract(full_contract())
    ids = [str(item["id"]) for item in base_contract["score_candidates"]]
    base = pd.DataFrame(
        {
            "id": ["well_a_10", "well_b_20"],
            "well": ["well_a", "well_b"],
            "well_row_idx": [10, 20],
            "outer_fold": np.array([0, 1], dtype=np.int8),
            "md_since": np.array([100.0, 1200.0], dtype=np.float32),
            "last_known_tvt": np.array([1000.0, 2000.0], dtype=np.float32),
        }
    )
    values = np.arange(24, dtype=np.float32).reshape(2, 12) + 1000.0
    return FoldBundle(
        base=base,
        values=values,
        available=np.ones_like(values, dtype=bool),
        confidence={},
        candidate_ids=ids,
        specs=contract_by_id(base_contract),
    )


def synthetic_pf_source() -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "id": ["well_a_10", "well_b_20"],
            "well": ["well_a", "well_b"],
            "row_idx": [10, 20],
            "pf_ancc_seed_mean_4": np.array([1001.0, 2001.0], dtype=np.float32),
            "pf_ancc_seed_mean_8": np.array([1002.0, 2002.0], dtype=np.float32),
            "pf_ancc_seed_std_4": np.array([1.0, 2.0], dtype=np.float32),
            "pf_ancc_seed_std_8": np.array([1.5, 2.5], dtype=np.float32),
            "pf_ancc_particle_std_mean_4": np.array([3.0, 4.0], dtype=np.float32),
            "pf_ancc_particle_std_mean_8": np.array([3.5, 4.5], dtype=np.float32),
            "pf_ancc_mean8_minus_mean4": np.array([1.0, 1.0], dtype=np.float32),
            "pf_ancc_mean8_minus_mean4_abs": np.array([1.0, 1.0], dtype=np.float32),
        }
    )
    return frame.set_index("id", drop=False)


@pytest.mark.parametrize(
    ("variant", "candidate_count", "primary_count", "disagreement"),
    [
        ("mean4_only", 12, 11, False),
        ("mean8_only", 12, 11, False),
        ("mean4_mean8_disagreement", 13, 12, True),
    ],
)
def test_variant_contract(
    variant: str, candidate_count: int, primary_count: int, disagreement: bool
) -> None:
    contract = build_variant_contract(full_contract(), variant)
    assert len(contract["score_candidates"]) == candidate_count
    assert len(contract["legal_domains"]["primitive_pair_bank"]["candidates"]) == primary_count
    assert contract["disagreement_enabled"] is disagreement
    assert contract["candidate_id_model_encoding"]["ordinal_index_as_model_feature"] is False
    assert "pf_ancc" not in [item["id"] for item in contract["score_candidates"]]
    selected = contract["variants"][variant]["external_candidates"]
    for domain in contract["legal_domains"].values():
        assert "pf_ancc" not in domain["candidates"]
        assert all(candidate in domain["candidates"] for candidate in selected)


def test_exp263_base_contract_retains_original_pf_ancc_only() -> None:
    contract = build_base_contract(full_contract())
    ids = [item["id"] for item in contract["score_candidates"]]
    assert len(ids) == 12
    assert ids[4] == "pf_ancc"
    assert not set(ids).intersection({"pf_ancc_seed_mean_4", "pf_ancc_seed_mean_8"})


def test_single_candidate_bundle_does_not_enable_disagreement() -> None:
    contract = build_variant_contract(full_contract(), "mean4_only")
    bundle = augment_fold_bundle(
        synthetic_base_bundle(contract), synthetic_pf_source(), contract
    )
    assert bundle.values.shape == (2, 12)
    replacement_position = bundle.candidate_ids.index("pf_ancc_seed_mean_4")
    assert replacement_position == 4
    assert "pf_ancc" not in bundle.candidate_ids
    assert np.array_equal(
        bundle.values[:, replacement_position],
        np.array([1001.0, 2001.0], np.float32),
    )
    confidence = bundle.confidence["pf_ancc_seed_mean_4"]
    assert not confidence["confidence_valid"].any()
    assert "pf_ancc_seed_std_4" not in confidence.columns


def test_both_candidate_bundle_adds_target_free_disagreement() -> None:
    contract = build_variant_contract(full_contract(), "mean4_mean8_disagreement")
    bundle = augment_fold_bundle(
        synthetic_base_bundle(contract), synthetic_pf_source(), contract
    )
    assert bundle.values.shape == (2, 13)
    assert bundle.candidate_ids[4:6] == [
        "pf_ancc_seed_mean_4",
        "pf_ancc_seed_mean_8",
    ]
    assert "pf_ancc" not in bundle.candidate_ids
    assert np.array_equal(
        bundle.values[:, 4:6],
        np.array([[1001.0, 1002.0], [2001.0, 2002.0]], np.float32),
    )
    for candidate in ["pf_ancc_seed_mean_4", "pf_ancc_seed_mean_8"]:
        confidence = bundle.confidence[candidate]
        assert confidence["confidence_valid"].all()
        assert np.array_equal(
            confidence["pf_ancc_mean8_minus_mean4_abs"].to_numpy(),
            np.array([1.0, 1.0], np.float32),
        )
        assert not any("target" in column.lower() for column in confidence.columns)
        assert not any("error" in column.lower() for column in confidence.columns)
        assert not any("oracle" in column.lower() for column in confidence.columns)


def test_candidate_join_fails_closed_on_row_mismatch() -> None:
    contract = build_variant_contract(full_contract(), "mean4_only")
    source = synthetic_pf_source().copy()
    source.loc["well_b_20", "row_idx"] = 21
    with pytest.raises(ValueError, match="row alignment"):
        augment_fold_bundle(synthetic_base_bundle(contract), source, contract)


def test_downstream_guard_requires_all_stress_checks() -> None:
    config = {
        "min_improved_folds": 3,
        "max_1000_plus_delta_rmse": 0.0,
        "max_hidden_like_delta_rmse": 0.0,
        "max_worst_well_regression": 0.25,
    }
    passed = downstream_guard(
        pooled_delta_rmse=-0.1,
        fold_deltas=[-0.1, -0.2, -0.1, 0.01, 0.02],
        distance_1000_plus_delta_rmse=-0.05,
        hidden_like_deltas=[-0.03, -0.02],
        worst_well_delta_rmse=0.2,
        guard_config=config,
    )
    assert passed["passed"] is True
    failed = downstream_guard(
        pooled_delta_rmse=-0.1,
        fold_deltas=[-0.1, -0.2, -0.1, 0.01, 0.02],
        distance_1000_plus_delta_rmse=-0.05,
        hidden_like_deltas=[-0.03, -0.02],
        worst_well_delta_rmse=0.3,
        guard_config=config,
    )
    assert failed["passed"] is False
    assert failed["checks"]["worst_well_regression_bounded"] is False


def test_corrected_exp264_mean4_run_has_exact_completed_compute_scope() -> None:
    config = yaml.safe_load((EXP / "config.yaml").read_text())
    assert config["experiment"]["status"] == (
        "corrected_mean4_nested_completed_score_guard_passed"
    )
    assert config["execution"]["stage"] == "nested_selector_mean4_only"
    assert config["execution"]["active_variant"] == "mean4_only"
    # The approved package is already on Kaggle; close the local gate to prevent a duplicate push.
    assert config["execution"]["run_approved"] is False
    assert config["execution"]["nested_selector_boosters_per_run"] == 40
    assert config["execution"]["downstream_boosters_per_run"] == 15
    assert config["execution"]["parent_control_retraining"] is False
    assert config["candidate_bank"]["replacement_target"] == "pf_ancc"
    assert config["candidate_bank"]["variants"]["mean4_only"][
        "expected_candidate_count"
    ] == 12
    assert config["candidate_bank"]["variants"]["mean8_only"][
        "expected_candidate_count"
    ] == 12
    assert config["candidate_bank"]["variants"]["mean4_mean8_disagreement"][
        "expected_candidate_count"
    ] == 13
    assert config["candidate_bank"]["hard_selection_enabled"] is False
    assert config["features"]["raw_context"]["horizontal_numeric_allowlist"] == [
        "MD",
        "X",
        "Y",
        "Z",
        "GR",
    ]
    assert config["features"]["raw_context"][
        "forbidden_training_only_columns"
    ] == ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]
    assert config["data"]["exp264_fixed_control_oof_sha256"] == (
        "7367983f3053186e0a6adf18c0f145302b0451332625fb679357f3c1326dafee"
    )
    assert config["data"]["exp264_fixed_control_expected_rmse"] == pytest.approx(
        10.476169179272501
    )
    downstream = config["model"]["downstream_tvt_stage"]
    assert downstream["expected_source_base_feature_count"] == 380
    assert downstream["expected_base_feature_count"] == 273
    assert downstream["expected_compact_feature_count"] == 74
    assert downstream["expected_final_feature_count"] == 347
    assert downstream["base_feature_allowlist_sha256"] == (
        "d01a73cc28485345dd86ed56ad6276f1727dca6b270d87685e1cf578afb677bf"
    )
    assert config["runtime"]["kaggle"]["kernel_sources"] == [
        "kentookumura/exp263-last-anchor-pair-cache-train",
        "kentookumura/exp271-pf-ancc-small-seed-mean-audit-train",
    ]
    assert config["inference"]["enabled"] is False
    assert config["inference"]["submit_to_kaggle"] is False


def test_raw_test_only_schema_guard_distinguishes_candidate_id_from_raw_formation() -> None:
    config = yaml.safe_load((EXP / "config.yaml").read_text())
    passed = raw_test_only_schema_guard(
        {
            "features": [
                "ctx__raw__md",
                "ctx__raw__gr",
                "id__candidate__pf_ancc_seed_mean_4",
            ]
        },
        config,
    )
    assert passed["passed"] is True
    assert passed["forbidden_feature_hits"] == []
    with pytest.raises(ValueError, match="training-only raw context"):
        raw_test_only_schema_guard(
            {"features": ["ctx__raw__ancc", "ctx__raw_delta_last__buda"]},
            config,
        )


def test_exp271_schema_contract_uses_object_strings() -> None:
    frame = pd.DataFrame(
        {
            "id": pd.Series(["well_a_1"], dtype=object),
            "well": pd.Series(["well_a"], dtype=object),
            "row_idx": np.array([1], dtype=np.int64),
            "pf_ancc_seed0": np.array([1.0], dtype=np.float32),
            "pf_ancc_seed_mean_4": np.array([1.0], dtype=np.float32),
            "pf_ancc_seed_mean_8": np.array([1.0], dtype=np.float32),
            "pf_ancc_seed_std_4": np.array([1.0], dtype=np.float32),
            "pf_ancc_seed_std_8": np.array([1.0], dtype=np.float32),
            "pf_ancc_particle_std_mean_4": np.array([1.0], dtype=np.float32),
            "pf_ancc_particle_std_mean_8": np.array([1.0], dtype=np.float32),
            "pf_ancc_mean8_minus_mean4": np.array([1.0], dtype=np.float32),
        }
    )
    assert schema_sha(frame) == (
        "9037c3e40cd7a4ad8535479dcad7ee16885c2214940a6c357915e8ec8b2a5ba9"
    )
