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
EXP_DIR = ROOT / "experiments" / "exp284_prefix_masked_wrong_mode_branch_recovery_backtest"
TRAIN_SOURCE = (
    EXP_DIR
    / "exp284_prefix_masked_wrong_mode_branch_recovery_backtest_compact_selfcontained_train.py"
)
INFERENCE_SOURCE = (
    EXP_DIR
    / "exp284_prefix_masked_wrong_mode_branch_recovery_backtest_compact_selfcontained_inference.py"
)


def load_module(path: Path = TRAIN_SOURCE, name: str = "exp284_train"):
    previous = os.environ.get("EXP284_IMPORT_ONLY")
    os.environ["EXP284_IMPORT_ONLY"] = "1"
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        if previous is None:
            os.environ.pop("EXP284_IMPORT_ONLY", None)
        else:
            os.environ["EXP284_IMPORT_ONLY"] = previous


def load_config() -> dict:
    return yaml.safe_load((EXP_DIR / "config.yaml").read_text())


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


def synthetic_proposals(control: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "well_id": ["well-a"] * 3,
            "fold": [0] * 3,
            "cut_row": [559] * 3,
            "event_row": [687] * 3,
            "control": [control] * 3,
            "proposal_rank": [1, 2, 3],
            "source": ["known_prefix"] * 3,
            "donor_row": [200, 300, 400],
            "orientation": ["forward", "reverse", "forward"],
            "anchor_tvt": [1138.0, 1148.0, 1158.0],
            "ncc17": [0.8, 0.7, 0.6],
            "ncc31": [0.8, 0.7, 0.6],
            "ncc51": [0.8, 0.7, 0.6],
            "multiscale_agreement": [1.0, 0.5, 0.0],
            "truth_attached": [False] * 3,
        }
    )


def test_config_and_zero_booster_contract() -> None:
    module = load_module(name="exp284_contract")
    config = load_config()
    module.validate_scientific_contract(config)
    assert config["experiment"]["route"] == "pf_beam"
    assert config["execution"]["active_backtest_variants"] == 1
    assert config["execution"]["fixed_policy_count"] == 5
    assert config["execution"]["lightgbm_config_count"] == 0
    assert config["execution"]["trained_fold_count"] == 0
    assert config["execution"]["total_boosters"] == 0
    assert config["execution"]["hmm_well_runs"] == 0
    assert config["execution"]["pf_well_runs"] == 0
    assert config["execution"]["control_or_parent_retraining"] is False
    assert config["execution"]["kaggle_push_approved"] is True
    assert config["dependency"]["current_status"] == "user_authorized_independent_execution"
    assert config["dependency"]["gate_override"]["authorized"] is True
    unauthorized = load_config()
    unauthorized["dependency"]["gate_override"]["authorized"] = False
    with pytest.raises(ValueError, match="override is not authorized"):
        module.validate_scientific_contract(unauthorized)


def test_mask_hides_exact_suffix_before_any_generator() -> None:
    module = load_module(name="exp284_mask")
    frame = synthetic_frame()
    masked, manifest = module.build_pseudo_mask("well-a", frame, load_config())
    assert manifest["original_last_known_row"] == 1199
    assert manifest["cut_row"] == 559
    assert manifest["event_row"] == 687
    assert manifest["masked_rows"] == 640
    assert masked.loc[:559, "TVT_input"].notna().all()
    assert masked.loc[560:, "TVT_input"].isna().all()
    assert manifest["post_cut_tvt_input_finite_rows_after_mask"] == 0

    with pytest.raises(ValueError, match="forbidden truth"):
        module.validate_target_safe_frame(masked.assign(TVT=np.arange(len(masked))))


def test_horizontal_loader_adds_deterministic_audit_ids(tmp_path: Path) -> None:
    module = load_module(name="exp284_horizontal_loader")
    path = tmp_path / "well-a__horizontal_well.csv"
    synthetic_frame(rows=8, known_rows=6).drop(columns="id").to_csv(path, index=False)

    loaded = module.load_target_safe_horizontal(path)

    assert loaded.columns.tolist() == ["id", "X", "Y", "Z", "MD", "GR", "TVT_input"]
    assert loaded["id"].tolist() == [f"well-a:{row_idx}" for row_idx in range(8)]
    module.validate_target_safe_frame(loaded)


def test_wrong_shift_selection_respects_local_maximum_and_minimum_offset() -> None:
    module = load_module(name="exp284_wrong_shift")
    shifts = [-80, -40, -20, -10, -5, -2, 0, 2, 5, 10, 20, 40, 80]
    scores = [-4, -3, 4, 1, 0, 0, 0, 0, 0, 2, 5, 1, -1]
    slot, used_local = module.select_wrong_shift(scores, shifts, 10.0)
    assert shifts[slot] == 20
    assert used_local is True
    assert abs(shifts[slot]) >= 10


def test_self_gr_proposal_is_causal_to_event_row() -> None:
    module = load_module(name="exp284_causal")
    config = load_config()
    frame = synthetic_frame()
    masked, manifest = module.build_pseudo_mask("well-a", frame, config)
    event = int(manifest["event_row"])
    emission = {"gr_fill": 70.0}
    first = module.build_self_gr_candidate_bank(
        "well-a",
        masked,
        cut=int(manifest["cut_row"]),
        event_row=event,
        emission=emission,
        config=config,
    )
    changed = masked.copy()
    changed.loc[event + 1 :, "GR"] = 1.0e6
    second = module.build_self_gr_candidate_bank(
        "well-a",
        changed,
        cut=int(manifest["cut_row"]),
        event_row=event,
        emission=emission,
        config=config,
    )
    columns = ["donor_row", "orientation", "ncc17", "ncc31", "ncc51"]
    pd.testing.assert_frame_equal(first[columns], second[columns])
    assert (first["source"] == "known_prefix").all()


def test_proposal_dedup_and_stable_shuffle_are_deterministic() -> None:
    module = load_module(name="exp284_proposal")
    config = load_config()
    candidates = pd.DataFrame(
        {
            "well_id": ["well-a"] * 6,
            "event_row": [687] * 6,
            "source": ["known_prefix"] * 6,
            "donor_row": [100, 110, 200, 300, 400, 500],
            "orientation": ["forward", "reverse", "forward", "forward", "reverse", "forward"],
            "anchor_tvt": [1000.0, 1010.0, 1040.0, 1080.0, 1120.0, 1160.0],
            "ncc17": [0.9, 0.89, 0.8, 0.7, 0.6, 0.5],
            "ncc31": [0.9, 0.89, 0.8, 0.7, 0.6, 0.5],
            "ncc51": [0.9, 0.89, 0.8, 0.7, 0.6, 0.5],
            "multiscale_agreement": [1.0, 1.0, 0.5, 0.5, 0.0, 0.0],
        }
    )
    real = module.rank_proposal_candidates(
        candidates, donor_dedup_rows=25, anchor_dedup_ft=2.0, top_k=3
    )
    assert real["donor_row"].tolist() == [100, 200, 300]
    first = module.stable_shuffled_proposals(
        real,
        candidates,
        well="well-a",
        event_row=687,
        seed=42,
        config=config,
    )
    second = module.stable_shuffled_proposals(
        real,
        candidates,
        well="well-a",
        event_row=687,
        seed=42,
        config=config,
    )
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 3


def test_future_evidence_and_policies_use_only_post_event_rows() -> None:
    module = load_module(name="exp284_evidence")
    config = load_config()
    frame = synthetic_frame()
    masked, manifest = module.build_pseudo_mask("well-a", frame, config)
    cut = int(manifest["cut_row"])
    event = int(manifest["event_row"])
    safe_geometry = 1112.0 + 0.2 * np.arange(1, 641)
    real = synthetic_proposals("real")
    shuffled = synthetic_proposals("shuffled").assign(
        donor_row=[210, 310, 410], anchor_tvt=[1140.0, 1150.0, 1160.0]
    )
    branches = module.build_branch_paths(
        "well-a",
        fold=0,
        cut=cut,
        event_row=event,
        safe_geometry=safe_geometry,
        wrong_shift=20.0,
        real_proposals=real,
        shuffled_proposals=shuffled,
        config=config,
    )
    typewell = pd.DataFrame({"TVT": np.linspace(900, 1400, 1001), "GR": np.linspace(40, 100, 1001)})
    emission = module.prepare_typewell_emission(masked, typewell, cut=cut, config=config)
    evidence = module.score_branch_evidence(branches, masked, emission, config)
    policies = module.select_fixed_policies(evidence)
    assert branches["row_idx"].min() == event + 1
    assert branches["future_offset"].min() == 1
    assert branches["future_offset"].max() == 512
    assert sorted(evidence["horizon_rows"].unique().tolist()) == [128, 256, 512]
    assert policies.groupby(["policy", "horizon_rows"]).size().eq(1).all()
    no_injection = policies.loc[
        policies["policy"] == "no_injection_base_plus_selfgr_top3",
        "selected_branch_id",
    ]
    assert not no_injection.eq("wrong_active").any()

    missing_a = masked.copy()
    missing_b = masked.copy()
    missing_a.loc[event + 120 : event + 128, "GR"] = np.nan
    missing_b.loc[event + 120 : event + 128, "GR"] = np.nan
    missing_a.loc[event + 129, "GR"] = -1.0e6
    missing_b.loc[event + 129, "GR"] = 1.0e6
    h128_a = module.score_branch_evidence(branches, missing_a, emission, config).loc[
        lambda frame: frame["horizon_rows"] == 128,
        ["branch_id", "likelihood_mean"],
    ]
    h128_b = module.score_branch_evidence(branches, missing_b, emission, config).loc[
        lambda frame: frame["horizon_rows"] == 128,
        ["branch_id", "likelihood_mean"],
    ]
    pd.testing.assert_frame_equal(h128_a.reset_index(drop=True), h128_b.reset_index(drop=True))

    mask_manifest = pd.DataFrame([manifest | {"fold": 0}])
    injection = pd.DataFrame(
        {
            "well_id": ["well-a"],
            "shift_ft": [20.0],
            "selected_wrong_shift": [True],
        }
    )
    proposals = pd.concat([real, shuffled], ignore_index=True)
    frozen = module.assert_target_free_tables(
        mask_manifest, injection, proposals, branches, evidence, policies
    )
    assert set(frozen) == {
        "mask_manifest",
        "injection",
        "proposals",
        "branch_paths",
        "evidence",
        "policy",
    }
    safe_truth = branches.loc[
        branches["branch_id"] == "safe_base", ["well_id", "row_idx", "branch_tvt"]
    ].rename(columns={"branch_tvt": "tvt_true"})
    readout = module.build_post_freeze_readout(
        branches, evidence, policies, safe_truth, pd.DataFrame()
    )
    assert len(readout["pooled_metrics"]) == 5 * 3 * 2
    assert bool(readout["safety"]["base_unique_best"].iloc[0])


def test_truth_reader_and_inference_fail_closed(tmp_path: Path) -> None:
    module = load_module(name="exp284_freeze")
    branch_paths = pd.DataFrame({"well_id": ["a"], "row_idx": [1]})
    with pytest.raises(ValueError, match="frozen content SHA"):
        module.load_truth_for_branch_rows(tmp_path, branch_paths, frozen_hashes={})

    inference = load_module(INFERENCE_SOURCE, "exp284_inference")
    config = load_config()
    inference.assert_inference_disabled(config)
    with pytest.raises(RuntimeError, match="train-side prefix-masked recovery backtest"):
        inference.fail_closed()
