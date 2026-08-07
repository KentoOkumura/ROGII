from __future__ import annotations

import copy
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = (
    ROOT
    / "experiments"
    / "exp302_exp226_multiscale_k_segment_candidate_audit"
)
TRAIN_SOURCE = (
    EXP_DIR
    / "exp302_exp226_multiscale_k_segment_candidate_audit_compact_selfcontained_train.py"
)
INFERENCE_SOURCE = (
    EXP_DIR
    / "exp302_exp226_multiscale_k_segment_candidate_audit_compact_selfcontained_inference.py"
)


def load_module(name: str = "exp302_train"):
    previous = os.environ.get("EXP302_IMPORT_ONLY")
    os.environ["EXP302_IMPORT_ONLY"] = "1"
    try:
        spec = importlib.util.spec_from_file_location(name, TRAIN_SOURCE)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        if previous is None:
            os.environ.pop("EXP302_IMPORT_ONLY", None)
        else:
            os.environ["EXP302_IMPORT_ONLY"] = previous


EXP302 = load_module()


@pytest.fixture
def config() -> dict:
    value = yaml.safe_load((EXP_DIR / "config.yaml").read_text())
    assert isinstance(value, dict)
    return value


def synthetic_horizontal(rows: int = 65, known_rows: int = 32) -> pd.DataFrame:
    index = np.arange(rows, dtype=np.float64)
    tvt = 1200.0 + 0.9 * index + 0.0002 * np.square(index)
    visible = tvt.copy()
    visible[known_rows:] = np.nan
    return pd.DataFrame(
        {
            "X": 1000.0 + 4.0 * index,
            "Y": 2000.0 + 2.0 * index,
            "Z": -1000.0 - index,
            "TVT": tvt,
            "TVT_input": visible,
            "ANCC": 300.0 + 0.2 * index,
            "GR": 80.0 + np.sin(index / 5.0),
        }
    )


def test_config_fixes_two_variants_ten_cpu_runs_and_zero_boosters(
    config: dict,
) -> None:
    assert config["experiment"]["status"] == (
        "kaggle_cpu_v2_completed_direct_fail_candidate_novelty_pass"
    )
    replay_config = copy.deepcopy(config)
    replay_config["experiment"]["status"] = "implementation_complete_not_executed"
    EXP302.validate_execution_contract(
        replay_config, require_kaggle_authorization=False
    )
    assert config["experiment"]["route"] == "pf_beam"
    assert config["execution"]["implementation"] is True
    assert config["execution"]["implementation_authorized"] is True
    assert config["execution"]["kaggle_execution_authorized"] is True
    assert config["execution"]["canonical_train_notebook_adopted"] is True
    assert config["execution"]["active_scientific_variants"] == 2
    assert config["execution"]["outer_evaluation_folds"] == 5
    assert config["execution"]["total_variant_fold_runs"] == 10
    assert config["execution"]["lightgbm_config_count"] == 0
    assert config["execution"]["trained_fold_count"] == 0
    assert config["execution"]["total_boosters"] == 0
    assert config["execution"]["parent_or_control_regeneration"] is False
    assert config["execution"]["one_run_authorization_consumed"] is True
    assert config["execution"]["technical_passed"] is True
    assert config["execution"]["direct_passed"] is False
    assert config["execution"]["candidate_novelty_passed"] is True
    assert config["execution"]["exp303_dependency_satisfied"] is True
    with pytest.raises(ValueError, match="status must identify"):
        EXP302.validate_execution_contract(
            config, require_kaggle_authorization=True
        )


def test_parameter_contract_changes_only_k_segments(config: dict) -> None:
    k12 = EXP302.params_for_variant(config, "exp226_k12")
    k24 = EXP302.params_for_variant(config, "exp226_k24")
    assert k12.k_segments == 12
    assert k24.k_segments == 24
    assert EXP302.replace(k12, k_segments=24) == k24
    changed = copy.deepcopy(config)
    changed["model"]["params"]["fixed_from_exp226"]["gate"] = 0.4
    with pytest.raises(ValueError, match="other than k_segments"):
        EXP302.params_for_variant(changed, "exp226_k12")


def test_segment_geometry_uses_exact_requested_resolution(config: dict) -> None:
    x = np.linspace(0.0, 500.0, 101)
    y = np.linspace(0.0, 100.0, 101)
    for variant, expected_k in EXP302.EXPECTED_VARIANTS.items():
        params = EXP302.params_for_variant(config, variant)
        segid, mid, projection, azimuth = EXP302.segment_geometry(
            x, y, 19, 81, params
        )
        assert len(mid) == expected_k
        assert len(projection) == expected_k
        assert len(azimuth) == expected_k
        assert int(segid.min()) == 0
        assert int(segid.max()) == expected_k - 1


def test_target_free_valid_object_and_donor_truth_attachment(
    tmp_path: Path, config: dict
) -> None:
    horizontal_path = tmp_path / "well-a__horizontal_well.csv"
    typewell_path = tmp_path / "well-a__typewell.csv"
    frame = synthetic_horizontal()
    frame.to_csv(horizontal_path, index=False)
    pd.DataFrame({"TVT": frame["TVT"], "GR": frame["GR"]}).to_csv(
        typewell_path, index=False
    )
    params = EXP302.params_for_variant(config, "exp226_k12")
    target_free = EXP302.load_target_free_wells(
        [horizontal_path], params
    )[0]
    assert target_free.tvt is None
    assert target_free.r0 is None
    assert target_free.c_raw is None
    assert target_free.c_sm is None
    donor = EXP302.attach_donor_fit_truth(
        target_free, horizontal_path, params
    )
    assert donor.tvt is not None
    assert donor.r0 is not None
    assert donor.c_raw is not None and len(donor.c_raw) == 12
    assert donor.c_sm is not None and len(donor.c_sm) == 12
    EXP302.assert_outer_fold_separation([], [target_free], 0)


def test_outer_valid_truth_state_is_rejected(tmp_path: Path, config: dict) -> None:
    path = tmp_path / "well-a__horizontal_well.csv"
    synthetic_horizontal().to_csv(path, index=False)
    params = EXP302.params_for_variant(config, "exp226_k24")
    target_free = EXP302.load_target_free_wells([path], params)[0]
    donor = EXP302.attach_donor_fit_truth(target_free, path, params)
    with pytest.raises(ValueError, match="outer-valid truth state"):
        EXP302.assert_outer_fold_separation([], [donor], 0)


def test_non_overlapping_block_assignment_includes_short_final_block() -> None:
    keys = pd.DataFrame(
        {
            "id": ["a_0", "a_1", "a_2", "b_0", "b_1"],
            "well": ["a", "a", "a", "b", "b"],
            "well_row_idx": [0, 1, 2, 0, 1],
            "outer_fold": [0, 0, 0, 1, 1],
            "md_since": [0.0, 1.0, 2.0, 0.0, 1.0],
        }
    )
    assignments = EXP302.build_block_assignments(keys, [2, 3])
    np.testing.assert_array_equal(
        assignments.frame["h2_group"].to_numpy(), [0, 0, 1, 2, 2]
    )
    np.testing.assert_array_equal(
        assignments.frame["h3_group"].to_numpy(), [0, 0, 0, 1, 1]
    )
    assert assignments.layouts["whole_well"].n_groups == 2


def test_exp226_evaluation_folds_are_separate_from_bank_provenance_folds() -> None:
    wells = [f"w{index}" for index in range(5) for _ in range(2)]
    bank_folds = np.repeat([1, 2, 3, 4, 0], 2)
    evaluation_folds = np.repeat([0, 1, 2, 3, 4], 2)
    keys = pd.DataFrame(
        {
            "id": [f"{well}_{index % 2}" for index, well in enumerate(wells)],
            "well": wells,
            "well_row_idx": np.tile([0, 1], 5),
            "outer_fold": bank_folds,
            "md_since": np.tile([0.0, 1.0], 5),
        }
    )
    aligned = pd.DataFrame(
        {
            "well_id": wells,
            "fold": evaluation_folds,
        }
    )
    bank = type("BankStub", (), {"keys": keys})()
    folds, fold_by_well, match_fraction = EXP302.extract_saved_control_folds(
        aligned, bank, 5
    )
    np.testing.assert_array_equal(folds, evaluation_folds)
    assert fold_by_well == {f"w{index}": index for index in range(5)}
    assert match_fraction == 0.0

    frozen = EXP302.build_block_assignments(keys, [2])
    evaluation = EXP302.assignments_with_evaluation_folds(frozen, folds)
    np.testing.assert_array_equal(frozen.well_fold, [1, 2, 3, 4, 0])
    np.testing.assert_array_equal(evaluation.well_fold, [0, 1, 2, 3, 4])
    np.testing.assert_array_equal(
        evaluation.layouts["h2"].group_fold, [0, 1, 2, 3, 4]
    )


def test_candidate_partition_rejects_truth_columns() -> None:
    EXP302.reject_forbidden_candidate_columns(["id", "candidate_tvt"])
    with pytest.raises(ValueError, match="forbidden truth"):
        EXP302.reject_forbidden_candidate_columns(
            ["id", "candidate_tvt", "tvt_true"]
        )


def direct_fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    records = []
    for variant, pooled in (
        ("exp226_k16", 9.427109674),
        ("exp226_k12", 9.30),
        ("exp226_k24", 9.50),
    ):
        records.append(
            {"variant": variant, "scope": "pooled", "fold": None, "rmse": pooled}
        )
        for fold in range(5):
            value = 9.0 + fold * 0.1
            if variant == "exp226_k12":
                value -= 0.1 if fold < 4 else -0.1
            elif variant == "exp226_k24":
                value += 0.2
            records.append(
                {"variant": variant, "scope": "fold", "fold": fold, "rmse": value}
            )
        for scope in (
            "1000_plus",
            "hidden_like_spatial",
            "hidden_like_typewell_purged",
        ):
            value = 10.0
            if variant == "exp226_k12":
                value += 0.01
            elif variant == "exp226_k24":
                value += 0.2
            records.append(
                {"variant": variant, "scope": scope, "fold": None, "rmse": value}
            )
    by_well = pd.DataFrame(
        [
            {"variant": variant, "well": f"w{idx}", "rmse": value}
            for variant, offset in (
                ("exp226_k16", 0.0),
                ("exp226_k12", 0.1),
                ("exp226_k24", 1.0),
            )
            for idx, value in enumerate(np.linspace(1.0, 10.0, 10) + offset)
        ]
    )
    return pd.DataFrame(records), by_well


def test_direct_guard_requires_all_registered_surfaces(config: dict) -> None:
    metrics, by_well = direct_fixture()
    result = EXP302.evaluate_direct_guards(metrics, by_well, config)
    assert result["passed"] is True
    assert result["variants"]["exp226_k12"]["passed"] is True
    assert result["variants"]["exp226_k12"]["improved_folds"] == 4
    assert result["variants"]["exp226_k24"]["passed"] is False


def novelty_fixture() -> pd.DataFrame:
    records = []
    for variant in EXP302.EXPECTED_VARIANTS:
        h512_gain = 0.04 if variant == "exp226_k12" else 0.01
        whole_gain = 0.03 if variant == "exp226_k12" else 0.01
        unique = 0.03 if variant == "exp226_k12" else 0.01
        records.extend(
            [
                {
                    "variant": variant,
                    "granularity": "h512",
                    "scope": "pooled",
                    "fold": None,
                    "base_oracle_rmse": 3.68,
                    "add_one_oracle_rmse": 3.68 - h512_gain,
                    "oracle_improvement_ft": h512_gain,
                    "strict_unique_best_fraction": unique,
                },
                {
                    "variant": variant,
                    "granularity": "whole_well",
                    "scope": "pooled",
                    "fold": None,
                    "base_oracle_rmse": 5.0,
                    "add_one_oracle_rmse": 5.0 - whole_gain,
                    "oracle_improvement_ft": whole_gain,
                    "strict_unique_best_fraction": unique,
                },
            ]
        )
        for fold in range(5):
            gain = h512_gain if fold < 4 else (-0.01 if variant == "exp226_k12" else 0.0)
            records.append(
                {
                    "variant": variant,
                    "granularity": "h512",
                    "scope": "fold",
                    "fold": fold,
                    "base_oracle_rmse": 3.7,
                    "add_one_oracle_rmse": 3.7 - gain,
                    "oracle_improvement_ft": gain,
                    "strict_unique_best_fraction": unique,
                }
            )
    return pd.DataFrame(records)


def test_candidate_novelty_guard_is_add_one_and_fold_strict(config: dict) -> None:
    result = EXP302.evaluate_novelty_guards(novelty_fixture(), config)
    assert result["passed"] is True
    assert result["variants"]["exp226_k12"]["passed"] is True
    assert result["variants"]["exp226_k12"]["improved_folds"] == 4
    assert result["variants"]["exp226_k24"]["passed"] is False


def test_novelty_tie_tolerance_keeps_frozen_bank_candidate() -> None:
    record = EXP302.novelty_metric_record(
        variant="exp226_k12",
        granularity="h512",
        scope="pooled",
        fold=None,
        group_mask=np.array([True, True]),
        group_rows=np.array([1, 1]),
        base_best_sse=np.array([1.0, 1.0]),
        added_sse=np.array([1.0 - 5.0e-10, 0.5]),
        tie_atol=1.0e-9,
    )
    assert record["strict_unique_best_groups"] == 1
    assert record["strict_unique_best_fraction"] == 0.5
    assert record["add_one_oracle_rmse"] == pytest.approx(np.sqrt(1.5 / 2.0))


def test_compact_source_is_not_a_thin_helper_notebook() -> None:
    source = TRAIN_SOURCE.read_text()
    assert "## 3. Exp226 deterministic geometry candidate helpers" in source
    assert "## 4. Exp293 fixed candidate-bank and block helpers" in source
    assert "## 5. Target-free OOF generation and freeze boundary" in source
    assert "## 6. Post-freeze truth loader and direct readout" in source
    assert "def build_variant_oof(" in source
    assert "def assignments_with_evaluation_folds(" in source
    assert "def build_novelty_readouts(" in source
    assert "bundle.values.astype(np.float32).astype(np.float64)" in source
    assert "__file__" not in source
    assert "from connortynan_k16_reproduction import" not in source
    assert "persist_oracle_prediction" not in source


def test_inference_candidate_is_fail_closed(config: dict) -> None:
    previous = os.environ.get("EXP302_IMPORT_ONLY")
    os.environ["EXP302_IMPORT_ONLY"] = "1"
    try:
        spec = importlib.util.spec_from_file_location(
            "exp302_inference", INFERENCE_SOURCE
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            os.environ.pop("EXP302_IMPORT_ONLY", None)
        else:
            os.environ["EXP302_IMPORT_ONLY"] = previous
    checks = module.validate_disabled_inference(config)
    assert all(checks.values())
    with pytest.raises(RuntimeError, match="intentionally disabled"):
        module.stop_without_inference(config)
