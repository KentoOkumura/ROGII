from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = (
    ROOT / "experiments" / "exp283_self_gr_topk_alternative_mode_branch_future_evidence_readout"
)
TRAIN_SOURCE = (
    EXP_DIR
    / (
        "exp283_self_gr_topk_alternative_mode_branch_future_evidence_readout_"
        "compact_selfcontained_train.py"
    )
)
INFERENCE_SOURCE = (
    EXP_DIR
    / (
        "exp283_self_gr_topk_alternative_mode_branch_future_evidence_readout_"
        "compact_selfcontained_inference.py"
    )
)


def load_module(path: Path = TRAIN_SOURCE, name: str = "exp283_train"):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_config() -> dict:
    value = yaml.safe_load((EXP_DIR / "config.yaml").read_text())
    assert isinstance(value, dict)
    return value


def test_fixed_zero_booster_contract() -> None:
    module = load_module(name="exp283_contract")
    config = load_config()
    module.validate_scientific_contract(config)
    assert config["experiment"]["route"] == "pf_beam"
    assert config["proposal"]["max_alternative_branches"] == 3
    assert config["evidence"]["primary_horizon_rows"] == 256
    assert config["execution"]["active_audit_variants"] == 1
    assert config["execution"]["lightgbm_config_count"] == 0
    assert config["execution"]["trained_fold_count"] == 0
    assert config["execution"]["total_boosters"] == 0
    assert config["execution"]["kaggle_push_approved"] is True
    assert config["execution"]["approval_source"] == "user_message_execute_2026_07_19"


def test_persistent_event_is_only_the_128th_contiguous_row() -> None:
    module = load_module(name="exp283_persistent")
    row_idx = np.r_[np.arange(100, 227), np.arange(300, 428)]
    mask = np.ones(len(row_idx), dtype=bool)
    events = module.persistent_run_event_rows(mask, row_idx, persistent_rows=128)
    assert events == [427]


def test_shift_margin_threshold_uses_other_folds_only() -> None:
    module = load_module(name="exp283_outer_train")
    margins = pd.DataFrame(
        {
            "well": [f"w{i}" for i in range(10)],
            "fold": np.repeat(np.arange(5), 2),
            "event_row_idx": np.arange(10),
            "shift_margin": np.arange(10, dtype=float),
        }
    )
    output = module.add_outer_train_margin_thresholds(
        margins, quantile=0.20, expected_folds=range(5)
    )
    expected_fold0 = margins.loc[margins["fold"] != 0, "shift_margin"].quantile(
        0.20, interpolation="linear"
    )
    assert output.loc[output["fold"] == 0, "shift_margin_outer_train_q20"].eq(expected_fold0).all()


def test_exp263_partition_is_independent_of_exp226_oof_fold() -> None:
    module = load_module(name="exp283_partition_identity")
    exp226 = pd.DataFrame(
        {
            "well_id": ["w0", "w0", "w1", "w1"],
            "row_idx": [0, 1, 0, 1],
            "suffix_offset": [0, 1, 0, 1],
            "fold": [0, 0, 1, 1],
            "tvt_geop": [1000.0, 1001.0, 1100.0, 1101.0],
        }
    )
    exp209 = pd.DataFrame(
        {
            "well": ["w0", "w0", "w1", "w1"],
            "row_idx": [0, 1, 0, 1],
            "md_since": [0.0, 1.0, 0.0, 1.0],
            "exact_hmm": [1000.0, 1001.0, 1100.0, 1101.0],
            "likpf_mean": [1000.0, 1001.0, 1100.0, 1101.0],
        }
    )
    exp236 = pd.DataFrame(
        {
            "id": ["w0_0", "w0_1", "w1_0", "w1_1"],
            "well": ["w0", "w0", "w1", "w1"],
            "row_idx": [0, 1, 0, 1],
            "bimodal_flag": [False] * 4,
        }
    )
    exp263 = pd.DataFrame(
        {
            "well_id": ["w0", "w0", "w1", "w1"],
            "row_idx": [0, 1, 0, 1],
            "base_fold": [1, 1, 0, 0],
            "base_tvt": [1000.0, 1001.0, 1100.0, 1101.0],
        }
    )
    config = load_config()
    config["validation"]["expected_rows"] = 4
    config["validation"]["expected_wells"] = 2
    config["validation"]["expected_folds"] = [0, 1]
    identity = module.build_target_free_identity(exp226, exp209, exp236, exp263, config)
    assert identity["fold"].tolist() == [0, 0, 1, 1]
    assert "base_fold" not in identity


def synthetic_proposal_inputs():
    rows = 800
    event_row = 500
    rng = np.random.default_rng(123)
    signal = rng.normal(0.0, 0.01, rows)
    motif = rng.normal(0.0, 1.0, 51)
    signal[30:81] = motif
    signal[450:501] = motif
    tvt_input = np.full(rows, np.nan)
    tvt_input[:100] = 1000.0 + 0.1 * np.arange(100)
    prepared = {
        "gr_smooth": signal,
        "tvt_input": tvt_input,
        "last_known_row": 99,
        "prediction_start_row": 100,
    }
    row_idx = np.arange(100, rows, dtype=np.int32)
    identity = pd.DataFrame(
        {
            "row_idx": row_idx,
            "base_tvt": 1000.0 + 0.1 * row_idx,
            "tvt_geop": 1000.0 + 0.1 * row_idx,
        }
    )
    event = {
        "event_id": "well_a_500",
        "well": "well_a",
        "fold": 0,
        "event_row_idx": event_row,
    }
    return event, prepared, identity


def test_causal_proposal_finds_forward_motif_and_shuffle_is_stable() -> None:
    module = load_module(name="exp283_proposal")
    event, prepared, identity = synthetic_proposal_inputs()
    first, manifest = module.build_proposals_for_event(event, prepared, identity, load_config())
    second, _ = module.build_proposals_for_event(event, prepared, identity, load_config())
    assert manifest["selected_proposals"] == 3
    assert int(first.loc[0, "donor_row_idx"]) == 80
    assert first.loc[0, "orientation"] == "forward"
    assert first.loc[0, "ncc51"] > 0.999
    assert first["shuffled_donor_row_idx"].equals(second["shuffled_donor_row_idx"])
    assert (first["proposal_window_end_row_idx"] < first["future_start_row_idx"]).all()
    assert len(module.assert_frozen_proposal_contract(first)) == 64


def synthetic_evidence_inputs(module):
    rows = 700
    event_row = 100
    true_tvt = 1000.0 + 0.1 * np.arange(rows)
    row_idx = np.arange(50, rows, dtype=np.int32)
    identity = pd.DataFrame(
        {
            "row_idx": row_idx,
            "base_tvt": true_tvt[row_idx] + 10.0,
            "tvt_geop": true_tvt[row_idx],
            "well_id": "well_a",
        }
    )
    typewell_tvt = np.linspace(900.0, 1200.0, 3001)
    prepared = {
        "gr": 2.0 * true_tvt,
        "typewell_tvt": typewell_tvt,
        "typewell_gr": 2.0 * typewell_tvt,
        "gr_sigma": 10.0,
        "prediction_start_row": 50,
        "last_known_row": 49,
        "tvt_input": np.r_[true_tvt[:50], np.full(rows - 50, np.nan)],
    }
    event = {
        "event_id": "well_a_100",
        "well": "well_a",
        "fold": 0,
        "event_row_idx": event_row,
    }
    proposal = pd.DataFrame(
        [
            {
                "event_id": "well_a_100",
                "well": "well_a",
                "fold": 0,
                "event_row_idx": event_row,
                "branch_rank": 1,
                "donor_source": "known_prefix",
                "orientation": "forward",
                "donor_row_idx": 40,
                "donor_anchor_tvt": true_tvt[event_row],
                "shuffled_donor_row_idx": 20,
                "shuffled_anchor_tvt": true_tvt[event_row] + 20.0,
                "ncc17": 1.0,
                "ncc31": 1.0,
                "ncc51": 1.0,
                "multiscale_agreement": 1.0,
                "base_event_tvt": true_tvt[event_row] + 10.0,
                "geop_event_tvt": true_tvt[event_row],
                "anchor_shift_ft": -10.0,
                "shuffled_anchor_shift_ft": 10.0,
                "proposal_window_end_row_idx": event_row,
                "future_start_row_idx": event_row + 1,
                "truth_attached": False,
            }
        ],
        columns=module.PROPOSAL_CONTENT_COLUMNS,
    )
    return event, proposal, prepared, identity, true_tvt


def test_future_evidence_selects_better_alternative_and_records_all_horizons() -> None:
    module = load_module(name="exp283_evidence")
    event, proposal, prepared, identity, _ = synthetic_evidence_inputs(module)
    evidence = module.build_future_evidence_for_event(
        event, proposal, prepared, identity, load_config()
    )
    assert sorted(evidence["horizon_rows"].unique()) == [128, 256, 512]
    selected = evidence.loc[
        (evidence["control"] == "real")
        & (evidence["horizon_rows"] == 256)
        & evidence["selected_primary"]
    ]
    assert len(selected) == 1
    assert int(selected["branch_rank"].iloc[0]) == 1
    assert len(module.assert_frozen_evidence_contract(evidence)) == 64


def test_postfreeze_boundary_and_readout() -> None:
    module = load_module(name="exp283_freeze")
    event, proposal, prepared, identity, true_tvt = synthetic_evidence_inputs(module)
    evidence = module.build_future_evidence_for_event(
        event, proposal, prepared, identity, load_config()
    )
    events = pd.DataFrame(
        [
            {
                "event_id": "well_a_100",
                "well": "well_a",
                "fold": 0,
                "event_row_idx": 100,
                "suffix_offset": 50,
                "md_since": 1000.0,
                "proposal_start_row_idx": 50,
                "future_start_row_idx": 101,
                "future_end_row_idx_h256": 356,
                "trigger_exp236_bimodal_segment_end": True,
                "trigger_exact_hmm_likpf_persistent": False,
                "trigger_exact_hmm_exp226_persistent": False,
                "trigger_low_shift_margin": False,
                "trigger_names": "exp236_bimodal_segment_end",
                "shift_margin": np.nan,
                "shift_margin_outer_train_q20": np.nan,
                "truth_attached": False,
            }
        ],
        columns=module.EVENT_CONTENT_COLUMNS,
    )
    event_sha = module.assert_frozen_event_contract(events)
    proposal_sha = module.assert_frozen_proposal_contract(proposal)
    evidence_sha = module.assert_frozen_evidence_contract(evidence)
    hidden = pd.DataFrame(
        {
            "well": ["well_a"],
            "verification_like_spatial_role": ["valid"],
            "verification_like_typewell_purged_role": ["train"],
        }
    )
    event_readout, pair_readout = module.build_postfreeze_readouts(
        events,
        proposal,
        evidence,
        identity,
        {"well_a": true_tvt},
        hidden,
        load_config(),
        event_sha=event_sha,
        proposal_sha=proposal_sha,
        evidence_sha=evidence_sha,
    )
    assert event_readout.loc[0, "selected_mse_h256"] < event_readout.loc[0, "base_mse_h256"]
    assert event_readout.loc[0, "selected_mse_h512"] < event_readout.loc[0, "base_mse_h512"]
    assert bool(event_readout.loc[0, "hidden_like_spatial"])
    assert not bool(event_readout.loc[0, "hidden_like_typewell_purged"])
    assert bool(pair_readout.loc[0, "alternative_better_than_base"])
    with pytest.raises(ValueError, match="64-character event"):
        module.load_truth_for_event_wells(
            Path("/does/not/matter"),
            ["well_a"],
            event_sha="",
            proposal_sha=proposal_sha,
            evidence_sha=evidence_sha,
        )


def test_truth_column_and_inference_fail_closed() -> None:
    module = load_module(name="exp283_leakage")
    event, _, _, _, _ = synthetic_evidence_inputs(module)
    unsafe = pd.DataFrame(
        [
            {
                **event,
                "suffix_offset": 0,
                "md_since": 1.0,
                "proposal_start_row_idx": 50,
                "future_start_row_idx": 101,
                "future_end_row_idx_h256": 356,
                **{
                    column: column == module.EVENT_TRIGGER_COLUMNS[0]
                    for column in module.EVENT_TRIGGER_COLUMNS
                },
                "trigger_names": "exp236_bimodal_segment_end",
                "shift_margin": np.nan,
                "shift_margin_outer_train_q20": np.nan,
                "truth_attached": False,
                "true_tvt": 1010.0,
            }
        ]
    )
    with pytest.raises(ValueError, match="forbidden truth"):
        module.assert_frozen_event_contract(unsafe)

    inference = load_module(INFERENCE_SOURCE, "exp283_inference")
    contract = inference.assert_inference_disabled(load_config())
    assert contract["boosters"] == 0
    with pytest.raises(RuntimeError, match="zero-booster"):
        inference.fail_closed()
