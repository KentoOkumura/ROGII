from __future__ import annotations

import copy
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
EXP = "exp433_rsd_sparse_anchor_direct_oof_readout"
EXP_DIR = ROOT / "experiments" / EXP
SOURCE = EXP_DIR / f"{EXP}_compact_selfcontained_train.py"


def load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def train() -> ModuleType:
    previous = os.environ.get("EXP433_IMPORT_ONLY")
    os.environ["EXP433_IMPORT_ONLY"] = "1"
    try:
        return load_module(SOURCE, "exp433_contract")
    finally:
        if previous is None:
            os.environ.pop("EXP433_IMPORT_ONLY", None)
        else:
            os.environ["EXP433_IMPORT_ONLY"] = previous


@pytest.fixture
def config() -> dict:
    value = yaml.safe_load((EXP_DIR / "config.yaml").read_text())
    assert isinstance(value, dict)
    return value


def make_score_bank(
    train: ModuleType,
    block_scores: list[dict[float, float]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    tie_priority = {float(offset): priority for priority, offset in enumerate(train.TIE_ORDER_FT)}
    for block_id, finite_scores in enumerate(block_scores):
        valid_slots = [
            slot for slot, offset in enumerate(train.OFFSETS_FT) if float(offset) in finite_scores
        ]
        ranked_slots = sorted(
            valid_slots,
            key=lambda slot: (
                -finite_scores[float(train.OFFSETS_FT[slot])],
                tie_priority[float(train.OFFSETS_FT[slot])],
            ),
        )
        ranked_slots.extend(slot for slot in range(13) if slot not in set(ranked_slots))
        ranks = np.empty(13, dtype=np.int64)
        for rank, slot in enumerate(ranked_slots, start=1):
            ranks[slot] = rank
        for slot, offset in enumerate(train.OFFSETS_FT):
            valid = slot in valid_slots
            score = finite_scores.get(float(offset), 0.0)
            row = {
                "well_id": "well-a",
                "fold": 0,
                "block_id": block_id,
                "block_start_suffix_offset": block_id * 512,
                "block_end_suffix_offset": block_id * 512 + 511,
                "block_start_row_idx": block_id * 512 + 10,
                "block_end_row_idx": block_id * 512 + 521,
                "block_row_count": 512,
                "md_since_min_ft": float(block_id * 512),
                "md_since_max_ft": float(block_id * 512 + 511),
                "md_since_mid_ft": float(block_id * 512 + 255.5),
                "raw_finite_gr_points": 512 if valid_slots else 0,
                "observed_gr_share": 1.0 if valid_slots else 0.0,
                "offset_slot": slot,
                "offset_ft": float(offset),
                "rsd_bin_score": float(score),
                "rsd_pearson": 0.0,
                "rsd_cosine": 0.0,
                "rsd_spearman": 0.0,
                "rsd_paired_bins": 32 if valid else 0,
                "rsd_valid": valid,
                "rsd_rank": int(ranks[slot]),
                "rsd_top3": bool(valid and ranks[slot] <= 3),
                "raw_pearson_score": 0.0,
                "raw_pearson_pairs": 0,
                "raw_pearson_valid": False,
                "raw_pearson_rank": slot + 1,
                "raw_pearson_top3": False,
                "raw_gaussian_score": 0.0,
                "raw_gaussian_valid": True,
                "raw_gaussian_rank": slot + 1,
                "raw_gaussian_top3": slot < 3,
                "permutation_score": 0.0,
                "permutation_valid": False,
                "permutation_rank": slot + 1,
                "permutation_top3": False,
            }
            rows.append(row)
    return pd.DataFrame(rows)[train.SCORE_LOGICAL_COLUMNS]


def test_contract_is_completed_and_remains_zero_model(
    train: ModuleType,
    config: dict,
) -> None:
    contract = train.validate_scientific_contract(config)
    assert contract["stage"] == "frozen_sparse_anchor_direct_oof_readout"
    assert contract["offsets_ft"] == train.OFFSETS_FT.tolist()
    assert contract["execution_counts"]["primary_decoders"] == 1
    assert contract["execution_counts"]["diagnostic_replays"] == 1
    for key in (
        "model_configs",
        "trained_folds",
        "boosters",
        "hmm_runs",
        "pf_runs",
        "beam_runs",
        "gpu_runs",
    ):
        assert contract["execution_counts"][key] == 0
    assert len(contract["scientific_contract_sha256"]) == 64

    approved = copy.deepcopy(config)
    approved["execution"]["run_train"] = True
    approved_contract = train.validate_scientific_contract(approved, require_run_approval=True)
    assert approved_contract["scientific_contract_sha256"] == contract["scientific_contract_sha256"]

    unapproved = copy.deepcopy(config)
    unapproved["execution"]["kaggle_package_authorized"] = False
    with pytest.raises(RuntimeError, match="not approved"):
        train.validate_scientific_contract(unapproved, require_run_approval=True)

    broken = copy.deepcopy(config)
    broken["execution_contract"]["score_regeneration"] = True
    with pytest.raises(ValueError, match="regeneration is forbidden"):
        train.validate_scientific_contract(broken)

    broken = copy.deepcopy(config)
    broken["data"]["exp426"]["score_bank"]["decompressed_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="frozen exp426 score_bank contract changed"):
        train.validate_scientific_contract(broken)


def test_frozen_score_structure_and_support_diagnostics_do_not_regenerate_scores(
    train: ModuleType,
    config: dict,
) -> None:
    score_bank = make_score_bank(
        train,
        [{0.0: 1.0, 2.0: 2.0}, {}, {-2.0: 3.0}],
    )
    checks = train.validate_score_bank_structure(score_bank)
    assert all(checks.values())

    local = copy.deepcopy(config)
    local["validation"]["expected_blocks"] = 3
    support, summary = train.build_support_diagnostics(score_bank, local)
    assert support["supported"].tolist() == [True, False, True]
    assert support["valid_offset_count"].tolist() == [2, 0, 1]
    assert support["zero_offset_valid"].tolist() == [True, False, False]
    assert support["symmetric_valid_pair_count"].tolist() == [0, 0, 0]
    assert summary["supported_block_fraction"] == pytest.approx(2 / 3)
    assert summary["coverage_is_gate"] is False
    np.testing.assert_array_equal(
        score_bank["rsd_bin_score"],
        make_score_bank(train, [{0.0: 1.0, 2.0: 2.0}, {}, {-2.0: 3.0}])["rsd_bin_score"],
    )


def test_viterbi_uses_transition_only_unsupported_block_and_fixed_hard_steps(
    train: ModuleType,
    config: dict,
) -> None:
    score_bank = make_score_bank(
        train,
        [
            {20.0: 100.0},
            {},
            {80.0: 100.0},
        ],
    )
    path = train.decode_well_viterbi(score_bank, config)
    assert path["selected_offset_ft"].tolist() == [20.0, 40.0, 80.0]
    assert path["supported"].tolist() == [True, False, True]
    assert path["selected_emission_score"].tolist() == [100.0, 0.0, 100.0]
    assert abs(path["selected_offset_ft"].iloc[0]) <= 20.0
    assert np.max(np.abs(np.diff(path["selected_offset_ft"]))) <= 40.0

    tied = train.choose_max_slot(np.ones(13, dtype=np.float64))
    assert train.OFFSETS_FT[tied] == 0.0


def test_row_projection_uses_fixed_centers_and_respects_slope_bound(
    train: ModuleType,
    config: dict,
) -> None:
    score_bank = make_score_bank(
        train,
        [
            {20.0: 100.0},
            {},
            {80.0: 100.0},
        ],
    )
    path = train.decode_well_viterbi(score_bank, config)
    suffix = np.arange(1536, dtype=np.int64)
    correction = train.project_datum_to_rows(
        suffix,
        path,
        block_size_rows=512,
    )
    assert correction[0] == 0.0
    assert correction[256] == pytest.approx(20.0)
    assert correction[768] == pytest.approx(40.0)
    assert correction[1280] == pytest.approx(80.0)
    assert np.max(np.abs(np.diff(correction))) <= 0.078125 + 1.0e-12


def test_prediction_freeze_requires_independent_full_and_probe_sha(
    train: ModuleType,
    config: dict,
) -> None:
    score_bank = make_score_bank(train, [{0.0: 1.0}, {2.0: 1.0}])
    rows = 1024
    safe = pd.DataFrame(
        {
            "well_id": "well-a",
            "row_idx": np.arange(10, 10 + rows, dtype=np.int64),
            "suffix_offset": np.arange(rows, dtype=np.int64),
            "fold": 0,
            "tvt_pred": np.arange(rows, dtype=np.float64),
        }
    )
    local = copy.deepcopy(config)
    local["validation"]["expected_rows"] = rows
    local["validation"]["expected_wells"] = 1
    local["validation"]["expected_blocks"] = 2
    local["validation"]["expected_score_rows"] = 26
    local["validation"]["fixed_probe_well"] = "well-a"
    datum, predictions, first = train.decode_all_wells(score_bank, safe, local)
    _, rerun, second = train.decode_all_wells(score_bank, safe, local)
    pd.testing.assert_frame_equal(predictions, rerun)

    support, _ = train.build_support_diagnostics(score_bank, local)
    well_manifest = pd.DataFrame({"well_id": ["well-a"], "blocks": [2]})
    input_manifest = pd.DataFrame({"well_id": ["well-a"], "horizontal_raw_sha256": ["a" * 64]})
    ledger = train.TruthAccessLedger()
    freeze = train.build_prediction_freeze(
        config=local,
        score_bank=score_bank,
        well_manifest=well_manifest,
        input_manifest=input_manifest,
        support=support,
        datum_path=datum,
        predictions=predictions,
        first_evidence=first,
        rerun_evidence=second,
        ledger=ledger,
    )
    assert all(freeze["checks"].values())
    assert ledger.frozen
    assert freeze["prediction_content_sha256"] == second["prediction_logical_sha256"]
    ledger.register_truth_access(rows)
    assert ledger.truth_rows_after_freeze == rows

    dirty = dict(second)
    dirty["prediction_logical_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="freeze failed"):
        train.build_prediction_freeze(
            config=local,
            score_bank=score_bank,
            well_manifest=well_manifest,
            input_manifest=input_manifest,
            support=support,
            datum_path=datum,
            predictions=predictions,
            first_evidence=first,
            rerun_evidence=dirty,
            ledger=train.TruthAccessLedger(),
        )


def test_truth_hidden_and_episode_access_fail_closed_before_freeze(
    train: ModuleType,
) -> None:
    ledger = train.TruthAccessLedger()
    with pytest.raises(RuntimeError, match="truth access"):
        ledger.register_truth_access(7)
    with pytest.raises(RuntimeError, match="hidden-role access"):
        ledger.register_hidden_role_access(3)
    with pytest.raises(RuntimeError, match="episode access"):
        ledger.register_episode_access(2)
    assert ledger.truth_rows_before_freeze == 7
    assert ledger.hidden_role_rows_before_freeze == 3
    assert ledger.episode_rows_before_freeze == 2


def test_episode_readout_uses_frozen_original_union_for_new_sse(
    train: ModuleType,
    config: dict,
) -> None:
    rows = 320
    base_error = np.zeros(rows, dtype=np.float64)
    primary_error = np.zeros(rows, dtype=np.float64)
    base_error[:128] = 12.0
    primary_error[:128] = 5.0
    primary_error[160:288] = 11.0
    evaluation = pd.DataFrame(
        {
            "well_id": "well-a",
            "fold": 0,
            "suffix_offset": np.arange(rows, dtype=np.int64),
            "base_error": base_error,
            "primary_error": primary_error,
        }
    )
    episodes = pd.DataFrame(
        {
            "well_id": ["well-a"],
            "fold": [0],
            "start_suffix_offset": [0],
            "end_suffix_offset_exclusive": [128],
            "rows": [128],
            "episode_sse": [128 * 12.0**2],
        }
    )
    detail, summary, mask = train.build_episode_readout(evaluation, episodes, config)
    assert mask.sum() == 128
    assert summary["persistent_episode_sse_reduction"] == pytest.approx(1.0 - 25.0 / 144.0)
    assert summary["persistent_episode_well_improvement_fraction"] == 1.0
    assert summary["new_corrected_episode_sse"] == pytest.approx(128 * 11.0**2)
    assert set(detail["episode_kind"]) == {
        "original_exp226",
        "corrected_detected",
    }


def test_prediction_metrics_routes_fold_and_by_well_to_dedicated_tables(
    train: ModuleType,
    config: dict,
) -> None:
    evaluation = pd.DataFrame(
        {
            "well_id": ["well-a", "well-a", "well-b", "well-b"],
            "fold": [0, 0, 1, 1],
            "base_error": [1.0, 2.0, 3.0, 4.0],
            "primary_error": [0.5, 1.5, 2.5, 3.5],
            "blockwise_error": [0.75, 1.75, 2.75, 3.75],
            "md_since_ft": [25.0, 75.0, 250.0, 1250.0],
            "raw_gr_missing": [False, True, False, True],
            "observed_gr_share": [1.0, 0.0, 0.75, 0.25],
            "hidden_like_spatial": [False, True, False, True],
            "hidden_like_typewell_purged": [True, False, True, False],
        }
    )
    scope_metrics, fold_metrics, by_well_metrics = train.build_prediction_metrics(
        evaluation,
        np.asarray([False, True, False, True]),
        config,
    )
    expected_scopes = set(config["validation"]["report_scopes"]) - {"fold", "by_well"}
    assert set(scope_metrics["scope"]) == expected_scopes
    assert "fold" not in set(scope_metrics["scope"])
    assert fold_metrics["fold"].tolist() == [0, 1]
    assert by_well_metrics["well_id"].tolist() == ["well-a", "well-b"]


def test_gate_is_fixed_and_fail_closed(
    train: ModuleType,
    config: dict,
) -> None:
    scopes = []
    for scope in config["validation"]["report_scopes"]:
        if scope in {"fold", "by_well"}:
            continue
        scopes.append(
            {
                "scope": scope,
                "rows": 100,
                "wells": 5,
                "base_rmse": 9.427109596582213,
                "primary_rmse": 9.20,
                "primary_gain_ft": 0.227109596582213,
                "primary_delta_rmse_ft": -0.227109596582213,
                "blockwise_rmse": 9.3,
                "blockwise_gain_ft": 0.127109596582213,
            }
        )
    folds = pd.DataFrame(
        {
            "fold": np.arange(5),
            "scope": "pooled",
            "rows": 20,
            "wells": 1,
            "base_rmse": 9.4,
            "primary_rmse": 9.2,
            "primary_gain_ft": 0.2,
            "primary_delta_rmse_ft": -0.2,
            "blockwise_rmse": 9.3,
            "blockwise_gain_ft": 0.1,
        }
    )
    by_well = pd.DataFrame(
        {
            "well_id": [f"well-{index}" for index in range(5)],
            "primary_delta_rmse_ft": np.full(5, -0.1),
        }
    )
    episode = {
        "persistent_episode_sse_reduction": 0.20,
        "persistent_episode_well_improvement_fraction": 0.70,
        "new_episode_sse_fraction_of_corrected_total": 0.01,
    }
    ledger = train.TruthAccessLedger()
    ledger.mark_frozen("a" * 64)
    gate, decision = train.evaluate_gates(
        config=config,
        input_evidence=[{"contract_checks": {"all": True}}],
        freeze={"checks": {"all": True}},
        first_decode={"maximum_row_correction_slope_ft": 0.078125},
        scope_metrics=pd.DataFrame(scopes),
        fold_metrics=folds,
        by_well_metrics=by_well,
        episode_summary=episode,
        ledger=ledger,
        elapsed_seconds=1.0,
        peak_memory_gb=0.1,
    )
    assert gate["all_passed"] is True
    assert decision["inference_enabled"] is False
    assert decision["submission_enabled"] is False
    assert decision["same_oof_rescue_allowed"] is False

    broken = pd.DataFrame(scopes)
    broken.loc[broken["scope"].eq("pooled"), "primary_gain_ft"] = 0.01
    gate, decision = train.evaluate_gates(
        config=config,
        input_evidence=[{"contract_checks": {"all": True}}],
        freeze={"checks": {"all": True}},
        first_decode={"maximum_row_correction_slope_ft": 0.078125},
        scope_metrics=broken,
        fold_metrics=folds,
        by_well_metrics=by_well,
        episode_summary=episode,
        ledger=ledger,
        elapsed_seconds=1.0,
        peak_memory_gb=0.1,
    )
    assert gate["scientific_passed"] is False
    assert decision["action"].startswith("scientific_fail_close")


def test_compact_candidate_is_adopted_as_canonical_train_only(
    config: dict,
) -> None:
    source = SOURCE.read_text()
    compact_notebook = EXP_DIR / f"{EXP}_compact_selfcontained_train.ipynb"
    canonical_train = (EXP_DIR / f"{EXP}_train.ipynb").read_text()
    canonical_inference = (EXP_DIR / f"{EXP}_inference.ipynb").read_text()
    assert config["experiment"]["status"] == "completed_scientific_failed_closed"
    assert config["implementation"]["enabled"] is True
    assert config["implementation"]["canonical_train_notebook_is_placeholder"] is False
    assert config["implementation"]["canonical_train_notebook_adopted"] is True
    assert config["implementation"]["canonical_inference_notebook_is_placeholder"] is True
    assert config["execution"]["canonical_notebook_replacement_authorized"] is True
    assert config["execution"]["kaggle_package_authorized"] is True
    assert config["execution"]["kaggle_push_authorized"] is True
    assert config["execution"]["kaggle_execution_authorized"] is True
    assert config["execution"]["kaggle_execution_completed"] is True
    assert config["execution"]["kaggle_kernel_version"] == 3
    assert config["execution"]["run_train"] is False
    assert config["execution"]["run_inference"] is False
    assert config["execution"]["create_submission"] is False
    assert compact_notebook.is_file()
    assert "__file__" not in source
    assert "decode_all_wells" in canonical_train
    assert "shutil.copyfile(sample_submission, submission_path)" in canonical_inference
