from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "experiments" / "exp291_prefix_masked_typewell_gr_multimode_safe_beam"
MODULE_PATH = EXP_DIR / (
    "exp291_prefix_masked_typewell_gr_multimode_safe_beam_compact_selfcontained_train.py"
)
os.environ["EXP291_IMPORT_ONLY"] = "1"
SPEC = importlib.util.spec_from_file_location("exp291_contract", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
EXP291 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = EXP291
SPEC.loader.exec_module(EXP291)


SHIFT_BANK = [-80.0, -40.0, -20.0, -10.0, -5.0, -2.0, 0.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0]


def load_config() -> dict:
    return yaml.safe_load((EXP_DIR / "config.yaml").read_text())


def minimal_config() -> dict:
    return {
        "pseudo_mask": {"masked_rows": 640},
        "candidate_bank": {"minimum_alternative_abs_shift_ft": 10.0},
        "policies": {"matched_shuffle": {"seed": 42}},
    }


def synthetic_frame(rows: int = 1500, known_rows: int = 1200) -> pd.DataFrame:
    index = np.arange(rows)
    tvt_input = 1000.0 + 0.2 * index
    tvt_input[known_rows:] = np.nan
    return pd.DataFrame(
        {
            "id": [f"id-{value}" for value in index],
            "X": 10.0 + 0.7 * index,
            "Y": 20.0 + 0.3 * index,
            "Z": -1000.0 - 0.1 * index,
            "MD": index.astype(float),
            "GR": 70.0 + 12.0 * np.sin(index / 17.0),
            "TVT_input": tvt_input,
        }
    )


def test_config_and_zero_booster_contract() -> None:
    config = load_config()
    EXP291.validate_scientific_contract(config)
    assert config["experiment"]["route"] == "pf_beam"
    assert config["execution"]["implementation"] is True
    assert config["execution"]["active_contract_count"] == 1
    assert config["execution"]["fixed_policy_count"] == 4
    assert config["execution"]["lightgbm_config_count"] == 0
    assert config["execution"]["trained_fold_count"] == 0
    assert config["execution"]["total_boosters"] == 0
    assert config["execution"]["hmm_well_runs"] == 0
    assert config["execution"]["pf_well_runs"] == 0
    assert config["execution"]["control_or_parent_retraining"] is False
    assert config["execution"]["kaggle_push_approved"] is True
    assert config["execution"]["canonical_train_notebook_adopted"] is True
    assert config["execution"]["canonical_inference_notebook_adopted"] is False

    unsafe = load_config()
    unsafe["candidate_bank"]["always_keep_safe_base"] = False
    with pytest.raises(ValueError, match="unpruned safe base"):
        EXP291.validate_scientific_contract(unsafe)


def test_mask_hides_exact_suffix_before_candidate_generation() -> None:
    frame = synthetic_frame()
    masked, manifest = EXP291.build_pseudo_mask("well-a", frame, load_config())
    assert manifest["original_last_known_row"] == 1199
    assert manifest["cut_row"] == 559
    assert manifest["masked_rows"] == 640
    assert masked.loc[:559, "TVT_input"].notna().all()
    assert masked.loc[560:, "TVT_input"].isna().all()
    assert manifest["post_cut_tvt_input_finite_rows_after_mask"] == 0
    with pytest.raises(ValueError, match="forbidden truth"):
        EXP291.validate_target_safe_frame(masked.assign(TVT=np.arange(len(masked))))


def test_horizontal_loader_adds_deterministic_audit_ids(tmp_path: Path) -> None:
    path = tmp_path / "well-a__horizontal_well.csv"
    synthetic_frame(rows=8, known_rows=6).drop(columns="id").to_csv(path, index=False)
    loaded = EXP291.load_target_safe_horizontal(path)
    assert loaded.columns.tolist() == ["id", "X", "Y", "Z", "MD", "GR", "TVT_input"]
    assert loaded["id"].tolist() == [f"well-a:{row_idx}" for row_idx in range(8)]
    EXP291.validate_target_safe_frame(loaded)


def shift_score_frame(local_slots: list[int]) -> pd.DataFrame:
    scores = np.linspace(-10.0, 2.0, len(SHIFT_BANK))
    scores[local_slots] = np.linspace(20.0, 10.0, len(local_slots))
    ranks = {slot: rank for rank, slot in enumerate(local_slots, start=1)}
    return pd.DataFrame(
        {
            "well_id": "well_a",
            "fold": 0,
            "cut_row": 1000,
            "shift_slot": np.arange(len(SHIFT_BANK)),
            "shift_ft": SHIFT_BANK,
            "visible_likelihood_mean": scores,
            "eligible_alternative": np.abs(SHIFT_BANK) >= 10.0,
            "eligible_local_maximum": [slot in local_slots for slot in range(len(SHIFT_BANK))],
            "real_mode_rank": pd.array(
                [ranks.get(slot, pd.NA) for slot in range(len(SHIFT_BANK))], dtype="Int16"
            ),
            "is_top1_real_mode": [
                bool(local_slots) and slot == local_slots[0]
                for slot in range(len(SHIFT_BANK))
            ],
            "truth_attached": False,
        }
    )


def test_local_maxima_have_no_forced_fallback() -> None:
    monotone_to_zero = [-6, -5, -4, -3, -2, -1, 0, -1, -2, -3, -4, -5, -6]
    assert EXP291.eligible_local_maximum_slots(monotone_to_zero, SHIFT_BANK, 10.0) == []
    scores = [-6, 4, -4, -3, -2, -1, 0, -1, -2, -3, 5, -5, 3]
    assert EXP291.eligible_local_maximum_slots(scores, SHIFT_BANK, 10.0) == [1, 10, 12]


def test_candidate_bank_keeps_safe_all_real_and_matched_count() -> None:
    frame = shift_score_frame([1, 10])
    first = EXP291.build_candidate_bank(
        "well_a", frame, fold=0, cut=1000, config=minimal_config()
    )
    second = EXP291.build_candidate_bank(
        "well_a", frame, fold=0, cut=1000, config=minimal_config()
    )
    assert first.equals(second)
    assert int((first["control"] == "core").sum()) == 1
    assert int((first["control"] == "real").sum()) == 2
    assert int((first["control"] == "shuffled").sum()) == 2
    assert set(first.loc[first["control"] == "real", "shift_slot"]) == {1, 10}


def test_candidate_bank_uses_safe_only_without_local_modes() -> None:
    frame = shift_score_frame([])
    bank = EXP291.build_candidate_bank(
        "well_a", frame, fold=0, cut=1000, config=minimal_config()
    )
    assert bank["branch_id"].tolist() == ["safe_base"]


def test_branch_paths_are_safe_plus_fixed_shift_for_512_rows() -> None:
    candidates = EXP291.build_candidate_bank(
        "well_a", shift_score_frame([1, 10]), fold=0, cut=1000, config=minimal_config()
    )
    safe = 1200.0 + np.arange(640, dtype=np.float64) * 0.2
    branches = EXP291.build_branch_paths(
        "well_a",
        fold=0,
        cut=1000,
        safe_geometry=safe,
        candidate_bank=candidates,
        config=minimal_config(),
    )
    assert branches["future_offset"].min() == 1
    assert branches["future_offset"].max() == 512
    assert branches["row_idx"].min() == 1001
    assert branches["row_idx"].max() == 1512
    for branch_id, part in branches.groupby("branch_id"):
        shift = float(part["shift_ft"].iloc[0])
        np.testing.assert_allclose(part["branch_tvt"].to_numpy(), safe[:512] + shift)
        if branch_id == "safe_base":
            assert shift == 0.0


def evidence_frame() -> pd.DataFrame:
    rows = []
    branches = [
        ("safe_base", "core", 0, False),
        ("real_mode_slot_01", "real", 2, True),
        ("real_mode_slot_10", "real", 11, False),
        ("shuffled_mode_slot_12", "shuffled", 112, False),
    ]
    values = {
        "safe_base": {128: 0.0, 256: 0.0, 512: 0.0},
        "real_mode_slot_01": {128: 1.0, 256: 1.5, 512: 2.0},
        "real_mode_slot_10": {128: 2.0, 256: -1.0, 512: 3.0},
        "shuffled_mode_slot_12": {128: -1.0, 256: 2.0, 512: 2.0},
    }
    for branch_id, control, order, top1 in branches:
        for horizon in EXP291.CHECKPOINTS:
            rows.append(
                {
                    "well_id": "well_a",
                    "fold": 0,
                    "cut_row": 1000,
                    "branch_id": branch_id,
                    "branch_type": "safe_base" if branch_id == "safe_base" else "mode",
                    "branch_order": order,
                    "control": control,
                    "source": "synthetic",
                    "shift_slot": 6 if branch_id == "safe_base" else order,
                    "shift_ft": 0.0 if branch_id == "safe_base" else 20.0,
                    "visible_rank": 0 if branch_id == "safe_base" else 1,
                    "is_top1_real_mode": top1,
                    "horizon_rows": horizon,
                    "likelihood_mean": values[branch_id][horizon],
                    "likelihood_sum": values[branch_id][horizon] * horizon,
                    "finite_path_and_steps": True,
                    "anchor_within_veto": True,
                    "typewell_support_with_extension": True,
                    "geometry_veto": False,
                    "truth_attached": False,
                }
            )
    return pd.DataFrame(rows)


def test_policy_requires_same_mode_to_persist() -> None:
    selected = EXP291.select_persistent_checkpoint_policies(evidence_frame())
    all_h256 = selected.loc[
        (selected["policy"] == "safe_plus_all_typewell_modes")
        & (selected["horizon_rows"] == 256)
    ].iloc[0]
    shuffled_h256 = selected.loc[
        (selected["policy"] == "safe_plus_matched_count_shuffled_modes")
        & (selected["horizon_rows"] == 256)
    ].iloc[0]
    assert all_h256["selected_branch_id"] == "real_mode_slot_01"
    assert shuffled_h256["selected_branch_id"] == "safe_base"
    assert int(all_h256["required_checkpoint_count"]) == 2


def test_inference_contract_is_explicitly_disabled() -> None:
    config = load_config()
    source = (
        EXP_DIR
        / "exp291_prefix_masked_typewell_gr_multimode_safe_beam_compact_selfcontained_inference.py"
    ).read_text()
    assert config["inference"]["enabled"] is False
    assert config["inference"]["create_submission"] is False
    assert "shutil.copyfile" not in source
    assert "decoder requires separate approval" in source
