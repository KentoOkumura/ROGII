from __future__ import annotations

from pathlib import Path

import yaml

from src.exact_datum_ranker_augmentation import (
    assign_stable_tvt_shifts,
    select_stable_wells,
)

ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = ROOT / "experiments" / "exp259_coordinate_equivariance_path_warp_augmentation"


def test_exp259_optional_train_compute_contract() -> None:
    config = yaml.safe_load((EXP_DIR / "config.yaml").read_text())
    assert config["execution"]["stage"] == "train_exact_datum_after_transform_audit"
    assert config["execution"]["model_training_enabled"] is True
    assert "train_exact_datum_after_transform_audit" in config["execution"]["allowed_stages"]
    assert config["model"]["active_variants"] == ["exact_tvt_datum_shift"]
    assert config["model"]["planned_lightgbm_configs"] == 2
    assert config["model"]["planned_folds"] == 5
    assert config["model"]["planned_boosters"] == 10
    assert config["model"]["control_retraining"] is False
    assert config["model"]["parent_retraining"] is False
    future = config["model"]["future_train_contract"]
    assert future["required_clean_control"] is False
    training = config["augmentation"]["training"]
    assert training["eligible_exact_transforms"] == ["tvt_datum_shift"]
    assert set(training["diagnostic_only_exact_transforms"]) == {
        "heel_center_translation",
        "lateral_reflection",
        "yaw_rotation",
    }
    assert "md_stretch" in training["disabled_transforms"]
    assert training["expected_selected_feature_count"] == 295
    assert training["expected_well_count"] == 773
    assert (
        training["selected_feature_schema_sha256"]
        == "7a9217d6ed96f5f1e569dbefff2a1fb17751405d6ddccae5e5d9dbf12da787ae"
    )


def test_augmented_well_and_shift_selection_is_stable() -> None:
    wells = [f"well_{index:03d}" for index in range(100)]
    left = select_stable_wells(wells, seed=42, namespace="exp259:fold3", fraction=0.25)
    right = select_stable_wells(
        list(reversed(wells)), seed=42, namespace="exp259:fold3", fraction=0.25
    )
    assert left == right
    assert len(left) == 25
    left_shift = assign_stable_tvt_shifts(
        left,
        seed=42,
        namespace="exp259:fold3",
        shift_grid_ft=[-40.0, -20.0, 20.0, 40.0],
    )
    right_shift = assign_stable_tvt_shifts(
        list(reversed(left)),
        seed=42,
        namespace="exp259:fold3",
        shift_grid_ft=[-40.0, -20.0, 20.0, 40.0],
    )
    assert left_shift == right_shift
    assert set(left_shift.values()) <= {-40.0, -20.0, 20.0, 40.0}
