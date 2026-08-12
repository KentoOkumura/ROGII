from __future__ import annotations

import importlib.util
import os
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = (
    ROOT
    / "experiments"
    / "exp354_typewell_group_candidate_family_prior_readout"
)
TRAIN_SOURCE = (
    EXP_DIR
    / "exp354_typewell_group_candidate_family_prior_readout_compact_selfcontained_train.py"
)
INFERENCE_SOURCE = EXP_DIR / (
    "exp354_typewell_group_candidate_family_prior_readout_"
    "compact_selfcontained_inference.py"
)


def load_module(path: Path, name: str):
    previous = os.environ.get("EXP354_IMPORT_ONLY")
    os.environ["EXP354_IMPORT_ONLY"] = "1"
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        if previous is None:
            os.environ.pop("EXP354_IMPORT_ONLY", None)
        else:
            os.environ["EXP354_IMPORT_ONLY"] = previous


@pytest.fixture(scope="module")
def train():
    return load_module(TRAIN_SOURCE, "exp354_train_test")


@pytest.fixture(scope="module")
def config():
    value = yaml.safe_load((EXP_DIR / "config.yaml").read_text())
    assert isinstance(value, dict)
    return value


def test_stage0_contract_is_completed_failed_closed_and_zero_model(train, config):
    contract = train.validate_scientific_contract(config)
    assert contract == {
        "prior_variants": 1,
        "negative_controls": 1,
        "reporting_folds": 5,
        "model_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
    }
    assert (
        config["experiment"]["status"]
        == "completed_stage_0_failed_close_without_rescue"
    )
    assert config["implementation"]["canonical_notebook_adopted"] is True
    assert config["implementation"]["stage_1_implemented"] is False
    assert config["execution"]["kaggle_push_approved"] is False
    assert config["execution"]["run_stage_0"] is False
    with pytest.raises(RuntimeError, match="package/push/run is not approved"):
        train.validate_run_approval(config)

    changed = deepcopy(config)
    changed["model"]["prior"]["shrinkage_support_k_wells"] = 5
    with pytest.raises(ValueError, match="scientific contract changed"):
        train.validate_scientific_contract(changed)


def test_candidate_family_manifest_is_fixed_and_complete(train, config):
    manifest = train.build_candidate_family_manifest(config)
    assert tuple(manifest["candidate_id"]) == train.EXPECTED_CANDIDATE_ORDER
    assert tuple(dict.fromkeys(manifest["family"])) == train.EXPECTED_FAMILY_ORDER
    assert manifest["candidate_position"].tolist() == list(range(12))
    assert manifest.loc[manifest["family"].eq("pf"), "candidate_id"].tolist() == [
        "likpf_mean",
        "pf_ancc",
    ]
    assert int(manifest["family"].eq("virtual_combination").sum()) == 6


def test_stable_shuffle_preserves_each_fold_multiset(train):
    membership = pd.DataFrame(
        {
            "well_id": [f"w{fold}_{slot}" for fold in range(5) for slot in range(4)],
            "outer_fold": [fold for fold in range(5) for _ in range(4)],
            "real_group_id": [
                f"g{slot % 3}" for _fold in range(5) for slot in range(4)
            ],
        }
    )
    first, first_sha = train.add_stable_group_label_shuffle(membership, 42)
    second, second_sha = train.add_stable_group_label_shuffle(
        membership.iloc[::-1], 42
    )
    pd.testing.assert_frame_equal(first, second)
    assert first_sha == second_sha
    for _fold, frame in first.groupby("outer_fold"):
        assert sorted(frame["real_group_id"]) == sorted(frame["shuffled_group_id"])
        assert frame["shuffle_offset"].nunique() == 1


def test_post_freeze_well_family_error_uses_equal_candidate_reducer(
    train, config, tmp_path
):
    local = deepcopy(config)
    local["data"]["exp293"]["expected_wells"] = 2
    family_manifest = train.build_candidate_family_manifest(local)
    raw_dir = tmp_path / "train"
    raw_dir.mkdir()
    block_rows = []
    for fold, well in enumerate(("wa", "wb")):
        pd.DataFrame(
            {
                "TVT": [0.0, 10.0, 20.0],
                "TVT_input": [0.0, np.nan, np.nan],
            }
        ).to_csv(raw_dir / f"{well}__horizontal_well.csv", index=False)
        block_rows.extend(
            [
                {
                    "id": f"{well}_1",
                    "well": well,
                    "well_row_idx": 1,
                    "outer_fold": fold,
                },
                {
                    "id": f"{well}_2",
                    "well": well,
                    "well_row_idx": 2,
                    "outer_fold": fold,
                },
            ]
        )
    block = pd.DataFrame(block_rows)
    values_path = tmp_path / "bank.f32"
    values = np.memmap(values_path, mode="w+", dtype="float32", shape=(4, 12))
    truth = np.array([10.0, 20.0, 10.0, 20.0], dtype=np.float32)
    values[:] = truth[:, None] + np.arange(12, dtype=np.float32)[None, :]
    values.flush()
    exp293 = train.Exp293Inputs(
        root=tmp_path,
        bank_manifest_path=tmp_path / "manifest.json",
        block_assignment_path=tmp_path / "block.csv.gz",
        candidate_bank_path=values_path,
        bank_manifest={"candidate_ids": list(train.EXPECTED_CANDIDATE_ORDER)},
        block_assignment=block,
        candidate_values=values,
        evidence={},
    )
    input_manifest = {
        "experiment": train.EXPERIMENT_NAME,
        "status": "target_free_identity_frozen_before_truth",
        "truth_rows_before_freeze": 0,
    }
    freeze_sha = train.stable_json_sha256(input_manifest)
    frozen = train.FrozenInputs(
        exp293=exp293,
        family_manifest=family_manifest,
        group_membership=pd.DataFrame(),
        raw_train_dir=raw_dir,
        input_manifest={**input_manifest, "target_free_freeze_sha256": freeze_sha},
        target_free_freeze_sha256=freeze_sha,
        truth_rows_before_freeze=0,
    )
    errors, evidence = train.compute_well_family_errors(frozen, local)
    geometry = errors.loc[
        errors["well_id"].eq("wa") & errors["family"].eq("geometry")
    ].iloc[0]
    pf = errors.loc[
        errors["well_id"].eq("wa") & errors["family"].eq("pf")
    ].iloc[0]
    assert geometry["rmse"] == pytest.approx(0.0)
    assert pf["mae"] == pytest.approx((2.0 + 4.0) / 2.0)
    assert pf["rmse"] == pytest.approx(np.sqrt((2.0**2 + 4.0**2) / 2.0))
    assert geometry["is_best_family"] == 1
    assert evidence["truth_attached_after_target_free_freeze"] is True


def synthetic_well_family_error(train):
    rows = []
    for fold in range(5):
        for slot in range(2):
            well = f"w{fold}_{slot}"
            group = slot
            for family_position, family in enumerate(train.EXPECTED_FAMILY_ORDER):
                rmse = 1.0 + family_position + 0.1 * group
                rows.append(
                    {
                        "well_id": well,
                        "outer_fold": fold,
                        "family": family,
                        "family_position": family_position,
                        "candidate_count": 1,
                        "suffix_rows": 2,
                        "mae": rmse,
                        "mse": rmse**2,
                        "rmse": rmse,
                        "is_best_family": int(family_position == 0),
                    }
                )
    return pd.DataFrame(rows)


def synthetic_membership():
    return pd.DataFrame(
        {
            "well_id": [f"w{fold}_{slot}" for fold in range(5) for slot in range(2)],
            "outer_fold": [fold for fold in range(5) for _ in range(2)],
            "real_group_id": [f"g{slot}" for _fold in range(5) for slot in range(2)],
            "shuffled_group_id": [
                f"g{1 - slot}" for _fold in range(5) for slot in range(2)
            ],
            "hidden_like_spatial_role": ["valid"] * 10,
            "hidden_like_typewell_purged_role": ["valid"] * 10,
        }
    )


def test_outer_train_prior_excludes_valid_wells_and_uses_fixed_shrinkage(
    train, config
):
    errors = synthetic_well_family_error(train)
    membership = synthetic_membership()
    schedule, freeze_sha = train.build_prior_schedule(errors, membership, config)
    assert len(freeze_sha) == 64
    assert schedule["fit_valid_well_overlap"].eq(0).all()
    assert schedule["outer_valid_truth_rows_before_prior_freeze"].eq(0).all()
    row = schedule.loc[
        schedule["control"].eq("real_native_group")
        & schedule["outer_fold"].eq(0)
        & schedule["well_id"].eq("w0_0")
        & schedule["family"].eq("geometry")
    ].iloc[0]
    assert row["source_wells"] == 4
    assert row["shrinkage_alpha"] == pytest.approx(4.0 / 14.0)
    assert row["selected_source"] == "group_family"


def test_perfect_real_rank_and_reversed_shuffle_pass_fixed_gate(train, config, tmp_path):
    local = deepcopy(config)
    local["data"]["exp293"]["expected_wells"] = 10
    errors = synthetic_well_family_error(train)
    membership = synthetic_membership()
    rows = []
    for control in train.CONTROLS:
        for record in errors.to_dict("records"):
            position = int(record["family_position"])
            prior_rmse = (
                float(position + 1)
                if control == "real_native_group"
                else float(len(train.EXPECTED_FAMILY_ORDER) - position)
            )
            rows.append(
                {
                    "control": control,
                    "outer_fold": record["outer_fold"],
                    "well_id": record["well_id"],
                    "group_id": "g",
                    "family": record["family"],
                    "family_position": position,
                    "selected_source": "group_family",
                    "fallback_reason": "",
                    "group_available": True,
                    "source_wells": 4,
                    "global_source_wells": 8,
                    "shrinkage_alpha": 4.0 / 14.0,
                    "prior_mae": prior_rmse,
                    "prior_mse": prior_rmse**2,
                    "prior_rmse": prior_rmse,
                    "prior_best_family_rate": 1.0 / (position + 1),
                    "fit_well_count": 8,
                    "fit_well_ids_sha256": "a" * 64,
                    "fit_valid_well_overlap": 0,
                    "outer_valid_truth_rows_before_prior_freeze": 0,
                }
            )
    schedule = pd.DataFrame(rows)
    freeze_sha = train.dataframe_content_sha256(schedule)
    schedule["prior_schedule_freeze_sha256"] = freeze_sha
    readout = train.attach_heldout_errors_after_freeze(
        schedule, freeze_sha, errors, membership
    )
    folds, surfaces = train.build_readout_metrics(readout, local)
    block = pd.DataFrame(
        {
            "well": [f"w{fold}_0" for fold in range(5)],
            "outer_fold": list(range(5)),
        }
    )
    exp293 = train.Exp293Inputs(
        root=tmp_path,
        bank_manifest_path=tmp_path / "manifest.json",
        block_assignment_path=tmp_path / "block.csv.gz",
        candidate_bank_path=tmp_path / "bank.f32",
        bank_manifest={"candidate_ids": list(train.EXPECTED_CANDIDATE_ORDER)},
        block_assignment=block,
        candidate_values=np.empty((0, 12), dtype=np.float32),
        evidence={},
    )
    family_manifest = train.build_candidate_family_manifest(local)
    frozen = train.FrozenInputs(
        exp293=exp293,
        family_manifest=family_manifest,
        group_membership=membership,
        raw_train_dir=tmp_path,
        input_manifest={},
        target_free_freeze_sha256="b" * 64,
        truth_rows_before_freeze=0,
    )
    gate = train.evaluate_stage_0_gate(
        frozen, schedule, readout, folds, surfaces, local
    )
    assert gate["passed"] is True
    assert gate["family_rank_spearman"] == pytest.approx(1.0)
    assert gate["shuffle_family_rank_spearman"] == pytest.approx(-1.0)
    assert gate["positive_folds"] == 5


def test_inference_is_fail_closed_and_sources_are_self_contained(train, config):
    inference = load_module(INFERENCE_SOURCE, "exp354_inference_test")
    contract = inference.validate_disabled_inference(config)
    assert contract["stage_0_boosters"] == 0
    assert contract["conditional_stage_1_models"] == 40
    with pytest.raises(RuntimeError, match="separate user approval"):
        inference.stop_disabled_inference(config)
    train_text = TRAIN_SOURCE.read_text()
    inference_text = INFERENCE_SOURCE.read_text()
    assert "def compute_well_family_errors" in train_text
    assert "def build_prior_schedule" in train_text
    assert "def evaluate_stage_0_gate" in train_text
    assert "def run_stage_0_experiment" in train_text
    assert "from settings import" not in train_text
    assert "from src" not in train_text
    assert "Path(__file__)" not in train_text
    assert "Path(__file__)" not in inference_text
