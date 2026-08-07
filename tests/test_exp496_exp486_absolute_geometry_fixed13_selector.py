from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.candidate_selector_pipeline import (
    KEY_COLUMNS,
    FoldBundle,
    compact_feature_names,
    read_yaml,
    validate_candidate_contract,
)
from src.exp486_fixed13_candidate_cache import (
    ADDED_CANDIDATE_ID,
    BASE_CANDIDATE_IDS,
    BASE_FIXED_IDS,
    BASE_PRIMARY_IDS,
    EXP486_LEDGER_FIELDS,
    Exp486Fixed13CandidateCache,
    base_exp264_contract,
    build_fixed13_integration_readout,
    build_incumbent_reranking_diagnostic,
    build_postfreeze_addone_novelty_readout,
    exp486_content_sha256,
    load_exp486_target_free_inputs,
    pair_selector_scores,
    resolve_csv_by_payload_sha,
    sha256_csv_payload,
    validate_fixed13_contract,
)

ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "experiments" / "exp496_exp486_absolute_geometry_fixed13_selector_on_exp264"


def contract() -> dict:
    return read_yaml(EXP_DIR / "candidate_contract.yaml")


def test_repository_config_has_exact_completed_stage_a_c_run_scope() -> None:
    config = read_yaml(EXP_DIR / "config.yaml")
    assert config["experiment"]["route"] == "ensemble"
    assert config["experiment"]["status"] == "completed_train_side_rejected_no_submit"
    assert config["implementation"]["approved"] is True
    assert config["implementation"]["code_created"] is True
    assert config["implementation"]["canonical_notebooks_are_placeholders"] is False
    assert config["implementation"]["canonical_train_notebook_adopted"] is True
    assert config["implementation"]["canonical_inference_notebook_is_placeholder"] is True
    assert config["implementation"]["kaggle_package_created"] is True
    assert config["execution"]["run_approved"] is True
    assert config["execution"]["approved_scope"] == (
        "fixed13_stage_a_plus_stage_c_1_variant_2_objectives_"
        "5_outer_4_inner_40_cpu_boosters_no_control_retraining"
    )
    assert config["execution"]["stage"] == (
        "stage_a_stage_c_completed_scientific_fail_terminal_close"
    )
    assert config["execution"]["trained_cpu_selector_boosters"] == 40
    assert config["execution"]["planned_cpu_boosters"] == 40
    assert config["execution"]["parent_control_retraining"] is False
    assert config["execution"]["candidate_pf_rerun_well_runs"] == 0
    assert config["execution"]["hmm_well_runs"] == 0
    assert config["execution"]["beam_well_runs"] == 0
    assert config["execution"]["gpu_boosters"] == 0
    assert config["execution"]["inference"] is False
    assert config["execution"]["submission"] is False
    assert config["execution"]["next_action"] == (
        "terminal_close_without_same_oof_rescue_or_inference"
    )
    assert config["results"]["decision"] == (
        "FAIL_CLOSE_EXP486_ABSOLUTE_FIXED13_SELECTOR"
    )
    assert config["results"]["scientific_gate_passed"] is False
    assert config["results"]["same_oof_rescue_allowed"] is False
    assert config["runtime"]["kaggle"]["status"] == "COMPLETE"


def test_fixed13_contract_is_single_exp486_add_one() -> None:
    value = contract()
    validate_fixed13_contract(value)
    assert [item["id"] for item in value["score_candidates"]] == [
        *BASE_CANDIDATE_IDS,
        ADDED_CANDIDATE_ID,
    ]
    assert tuple(value["legal_domains"]["primitive_pair_bank"]["candidates"]) == (
        *BASE_PRIMARY_IDS,
        ADDED_CANDIDATE_ID,
    )
    assert tuple(value["legal_domains"]["primitive_fixed_bank"]["candidates"]) == BASE_FIXED_IDS
    assert len(compact_feature_names(value)) == 77
    assert value["added_candidate_contract"]["new_pair_or_blend_candidates"] == []


def test_base_contract_remains_valid_exp264_fixed12() -> None:
    base = base_exp264_contract(contract())
    validate_candidate_contract(base)
    assert [item["id"] for item in base["score_candidates"]] == list(BASE_CANDIDATE_IDS)


def _write_exp486_source(tmp_path: Path) -> tuple[Path, Path, Path, dict[str, str]]:
    prediction = pd.DataFrame(
        {
            "id": ["a_1", "a_2", "b_3"],
            "well_id": ["a", "a", "b"],
            "row_idx": [1, 2, 3],
            "suffix_offset": [0, 1, 0],
            "likpf_scale5_absolute_geometry_unary": [10.0, 11.0, 20.0],
            "likpf_scale5_slow_residual_offset": [100.0, 100.0, 100.0],
            "true_tvt": [9.0, 9.0, 19.0],
        }
    )
    ledger = pd.DataFrame(
        {
            "id": ["a_1", "a_2", "b_3"],
            "well_id": ["a", "a", "b"],
            "row_idx": [1, 2, 3],
            "suffix_offset": [0, 1, 0],
            "tvt_geop": [10.0, 11.0, 20.0],
            "geometry_residual_mean": [0.1, 0.2, 0.3],
            "geometry_residual_std": [1.0, 2.0, 3.0],
            "geometry_log_factor_mean": [-1.0, -2.0, -3.0],
            "effective_sample_size": [100.0, 90.0, 80.0],
            "resampled_seed_fraction": [0.0, 0.5, 1.0],
            "TVT": [9.0, 9.0, 19.0],
        }
    )
    prediction_path = tmp_path / "predictions.csv.gz"
    ledger_path = tmp_path / "absolute.csv.gz"
    prediction.to_csv(prediction_path, index=False, compression="gzip")
    ledger.to_csv(ledger_path, index=False, compression="gzip")
    prediction_payload = sha256_csv_payload(prediction_path)
    ledger_payload = sha256_csv_payload(ledger_path)
    logical = "a" * 64
    freeze = {
        "frozen_before_truth_attachment": True,
        "rows": 3,
        "wells": 2,
        "scientific_contract_sha256": "b" * 64,
        "prediction_logical_sha256": logical,
        "absolute_ledger_logical_sha256": ledger_payload,
        "sha_readback": {"pass": True},
    }
    freeze_path = tmp_path / "freeze.json"
    freeze_path.write_text(json.dumps(freeze))
    return (
        prediction_path,
        ledger_path,
        freeze_path,
        {
            "prediction_raw": hashlib.sha256(prediction_path.read_bytes()).hexdigest(),
            "prediction_payload": prediction_payload,
            "prediction_logical": logical,
            "ledger_raw": hashlib.sha256(ledger_path.read_bytes()).hexdigest(),
            "ledger_payload": ledger_payload,
            "freeze_raw": hashlib.sha256(freeze_path.read_bytes()).hexdigest(),
        },
    )


def test_exp486_loader_uses_two_strict_allowlists_and_freeze_manifest(
    tmp_path: Path,
) -> None:
    prediction_path, ledger_path, freeze_path, sha = _write_exp486_source(tmp_path)
    loaded, manifest = load_exp486_target_free_inputs(
        prediction_path,
        ledger_path,
        freeze_path,
        expected_rows=3,
        expected_wells=2,
        expected_prediction_gzip_raw_sha256=sha["prediction_raw"],
        expected_prediction_payload_sha256=sha["prediction_payload"],
        expected_prediction_upstream_logical_sha256=sha["prediction_logical"],
        expected_ledger_gzip_raw_sha256=sha["ledger_raw"],
        expected_ledger_payload_sha256=sha["ledger_payload"],
        expected_freeze_manifest_sha256=sha["freeze_raw"],
        expected_scientific_contract_sha256="b" * 64,
        expected_exp226_geometry_decompressed_sha256="c" * 64,
    )
    assert list(loaded.columns) == [
        "id",
        "well_id",
        "row_idx",
        "suffix_offset",
        "candidate_tvt",
        *EXP486_LEDGER_FIELDS,
    ]
    assert manifest["forbidden_truth_error_control_role_fold_scope_gate_columns_loaded"] == 0
    assert "true_tvt" in manifest["prediction_header_columns"]
    assert "TVT" in manifest["absolute_ledger_header_columns"]
    assert manifest["upstream_fold_column_loaded"] is False
    assert manifest["upstream_fold_used_as_model_feature"] is False
    assert manifest["candidate_and_native_confidence_finite_fraction"] == 1.0
    assert manifest["post_read_content_sha256"] == exp486_content_sha256(loaded)


def test_csv_resolver_accepts_plain_unpacked_payload(tmp_path: Path) -> None:
    path = tmp_path / "source.csv"
    path.write_bytes(b"a,b\n1,2\n")
    payload_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    resolved = resolve_csv_by_payload_sha(
        ["**/source.csv"],
        [tmp_path],
        expected_payload_sha256=payload_sha,
        expected_gzip_raw_sha256="f" * 64,
        label="unpacked Kaggle dataset payload",
    )
    assert resolved == path


class _FakeBaseCache:
    def __init__(self, bundles: FoldBundle | dict[int, FoldBundle]):
        self.bundles = bundles

    def load_fold(self, fold: int) -> FoldBundle:
        if isinstance(self.bundles, dict):
            return self.bundles[fold]
        return self.bundles


def _base_bundle(fold: int, rows: list[tuple[str, int]]) -> FoldBundle:
    base = pd.DataFrame(
        {
            "id": [f"{well}_{row}" for well, row in rows],
            "well": [well for well, _ in rows],
            "well_row_idx": [row for _, row in rows],
            "outer_fold": [fold] * len(rows),
            "md_since": np.arange(len(rows), dtype=np.float32),
            "last_known_tvt": [100.0] * len(rows),
        }
    )
    base_values = np.arange(len(rows) * 12, dtype=np.float32).reshape(len(rows), 12)
    confidence = {
        name: base[KEY_COLUMNS].assign(confidence_valid=True) for name in BASE_CANDIDATE_IDS[:6]
    }
    return FoldBundle(
        base=base,
        values=base_values,
        available=np.ones_like(base_values, dtype=bool),
        confidence=confidence,
        candidate_ids=list(BASE_CANDIDATE_IDS),
        specs={
            str(item["id"]): dict(item)
            for item in base_exp264_contract(contract())["score_candidates"]
        },
    )


def test_fixed13_cache_appends_exp486_in_base_row_order() -> None:
    base = _base_bundle(0, [("b", 6), ("a", 2), ("a", 3)])
    source = pd.DataFrame(
        {
            "well_id": ["a", "a", "b"],
            "row_idx": [2, 3, 6],
            "candidate_tvt": [202.0, 203.0, 206.0],
            "suffix_offset": [0, 1, 0],
            "geometry_residual_mean": [2.0, 3.0, 6.0],
            "geometry_residual_std": [2.0, 3.0, 6.0],
            "geometry_log_factor_mean": [-2.0, -3.0, -6.0],
            "effective_sample_size": [20.0, 30.0, 60.0],
            "resampled_seed_fraction": [0.2, 0.3, 0.6],
        }
    )
    cache = Exp486Fixed13CandidateCache.__new__(Exp486Fixed13CandidateCache)
    cache.contract = contract()
    cache.ids = [*BASE_CANDIDATE_IDS, ADDED_CANDIDATE_ID]
    cache.specs = {str(item["id"]): dict(item) for item in cache.contract["score_candidates"]}
    cache.base_cache = _FakeBaseCache(base)
    cache.exp486_manifest = {
        "upstream_exp226_group_safe_oof_contract": True,
        "upstream_fold_column_loaded": False,
        "upstream_fold_used_as_model_feature": False,
    }
    cache.exp486_by_key = source.set_index(["well_id", "row_idx"])[
        ["candidate_tvt", "suffix_offset", *EXP486_LEDGER_FIELDS]
    ]
    cache._selector_fold_audits = {}
    extended = cache.load_fold(0)
    assert extended.values[:, -1].tolist() == [206.0, 202.0, 203.0]
    added_confidence = extended.confidence[ADDED_CANDIDATE_ID]
    assert added_confidence["confidence_valid"].all()
    assert added_confidence["geometry_residual_mean"].tolist() == [6.0, 2.0, 3.0]
    assert added_confidence["effective_sample_size"].tolist() == [60.0, 20.0, 30.0]


def test_selector_repartition_audits_all_five_folds() -> None:
    bundles: dict[int, FoldBundle] = {}
    rows = []
    for fold in range(5):
        well = f"w{fold}"
        row_idx = 10 + fold
        bundles[fold] = _base_bundle(fold, [(well, row_idx)])
        rows.append(
            {
                "well_id": well,
                "row_idx": row_idx,
                "candidate_tvt": 200.0 + fold,
                "suffix_offset": 0,
                "geometry_residual_mean": 0.0,
                "geometry_residual_std": 1.0,
                "geometry_log_factor_mean": -1.0,
                "effective_sample_size": 50.0,
                "resampled_seed_fraction": 0.5,
            }
        )
    cache = Exp486Fixed13CandidateCache.__new__(Exp486Fixed13CandidateCache)
    cache.contract = contract()
    cache.ids = [*BASE_CANDIDATE_IDS, ADDED_CANDIDATE_ID]
    cache.specs = {str(item["id"]): dict(item) for item in cache.contract["score_candidates"]}
    cache.base_cache = _FakeBaseCache(bundles)
    cache.exp486_manifest = {
        "upstream_exp226_group_safe_oof_contract": True,
        "upstream_fold_column_loaded": False,
        "upstream_fold_used_as_model_feature": False,
    }
    cache.exp486_by_key = pd.DataFrame(rows).set_index(["well_id", "row_idx"])[
        ["candidate_tvt", "suffix_offset", *EXP486_LEDGER_FIELDS]
    ]
    cache._selector_fold_audits = {}
    for fold in range(5):
        cache.load_fold(fold)
    audit = cache.selector_repartition_manifest(expected_rows=5)
    assert audit["passed"] is True
    assert audit["upstream_source_fold_loaded"] is False
    assert audit["upstream_source_fold_used_as_model_feature"] is False


def _score_frame(*, fixed13: bool) -> pd.DataFrame:
    ids = (*BASE_CANDIDATE_IDS, ADDED_CANDIDATE_ID) if fixed13 else BASE_CANDIDATE_IDS
    rows: list[dict] = []
    for fold in range(5):
        well = f"w{fold}"
        for row_idx, md_since in ((0, 100.0), (1, 1100.0)):
            for position, candidate in enumerate(ids):
                actual = 1.0 if candidate in {"exp226_k16", "exact_hmm"} else 3.0
                predicted = 0.5 + position
                if fixed13 and candidate == ADDED_CANDIDATE_ID:
                    actual = 0.5
                if fixed13 and row_idx == 0 and candidate == ADDED_CANDIDATE_ID:
                    predicted = 0.1
                if fixed13 and row_idx == 1 and candidate == "exact_hmm":
                    predicted = 0.1
                if candidate == "exp226_w500_50_50":
                    actual = 3.0
                rows.append(
                    {
                        "id": f"{well}_{row_idx}",
                        "well": well,
                        "well_row_idx": row_idx,
                        "outer_fold": fold,
                        "md_since": md_since,
                        "candidate_id": candidate,
                        "candidate_tvt": 100.0 + position,
                        "candidate_available": True,
                        "confidence_valid": True,
                        "actual_abs_error": actual,
                        "actual_within10": 1,
                        "pred_abs_error": predicted,
                        "p_within10": 1.0 / (1.0 + predicted),
                    }
                )
    return pd.DataFrame(rows)


def _write_readout_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    new_path = tmp_path / "new.parquet"
    parent_path = tmp_path / "parent.parquet"
    _score_frame(fixed13=True).to_parquet(new_path, index=False)
    _score_frame(fixed13=False).to_parquet(parent_path, index=False)
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    for fold in range(5):
        well = f"w{fold}"
        pd.DataFrame({"GR": [10.0, np.nan]}).to_csv(
            raw_dir / f"{well}__horizontal_well.csv", index=False
        )
    assignment = pd.DataFrame(
        {
            "well_id": [f"w{fold}" for fold in range(5)],
            "verification_like_spatial_role": ["valid"] * 5,
            "verification_like_typewell_purged_role": ["valid"] * 5,
        }
    )
    assignment_path = tmp_path / "assignment.csv"
    assignment.to_csv(assignment_path, index=False)
    return new_path, parent_path, raw_dir, assignment_path


def test_paired_integration_gate_covers_all_seven_scopes(tmp_path: Path) -> None:
    new_path, parent_path, raw_dir, assignment_path = _write_readout_inputs(tmp_path)
    paired = pair_selector_scores(
        new_score_path=new_path,
        parent_score_path=parent_path,
        contract=contract(),
    )
    guard_config = {
        "minimum_added_candidate_primary_top1_fraction": 0.005,
        "minimum_positive_usage_folds": 4,
        "maximum_pooled_delta_rmse_vs_parent_fixed12_selector": 0.0,
        "minimum_improved_folds_vs_parent_fixed12_selector": 4,
        "maximum_scope_delta_rmse_ft": 0.02,
        "high_missing_fraction_threshold": 0.30,
        "required_scopes": [
            "raw_gr_observed",
            "raw_gr_missing",
            "missing_fraction_high",
            "distance_0_250",
            "distance_1000_plus",
            "hidden_like_spatial",
            "hidden_like_typewell_purged",
        ],
        "maximum_by_well_p95_delta_rmse_ft": 0.25,
        "maximum_worst_well_delta_rmse_ft": 0.25,
        "failure_decision": "FAIL_CLOSE_EXP486_ABSOLUTE_FIXED13_SELECTOR",
    }
    gate, by_well = build_fixed13_integration_readout(
        paired=paired,
        hidden_like_assignment_path=assignment_path,
        raw_train_dir=raw_dir,
        score_summary={"score_guard": {"passed": True}},
        guard_config=guard_config,
        output_dir=tmp_path / "out",
    )
    assert gate["passed"] is True
    assert gate["positive_exp486_usage_folds"] == 5
    assert gate["fixed_fallback_error_parity_max_abs_ft"] == 0.0
    assert by_well["exp486_top1_fraction"].eq(0.5).all()
    assert (tmp_path / "out" / "exp496_scientific_gate.json").exists()

    failed_gate, _ = build_fixed13_integration_readout(
        paired=paired,
        hidden_like_assignment_path=assignment_path,
        raw_train_dir=raw_dir,
        score_summary={"score_guard": {"passed": False}},
        guard_config=guard_config,
        output_dir=tmp_path / "failed_score_guard",
    )
    assert failed_gate["passed"] is False
    assert failed_gate["checks"]["selector_score_guard"] is False
    assert failed_gate["decision"] == "FAIL_CLOSE_EXP486_ABSOLUTE_FIXED13_SELECTOR"
    assert (tmp_path / "failed_score_guard" / "exp496_scientific_gate.json").exists()


def test_postfreeze_novelty_and_reranking_are_diagnostic_only(tmp_path: Path) -> None:
    new_path, parent_path, _, _ = _write_readout_inputs(tmp_path)
    paired = pair_selector_scores(
        new_score_path=new_path,
        parent_score_path=parent_path,
        contract=contract(),
    )
    by_well_rows = []
    for well, part in paired.groupby("well", sort=True):
        added = part["new_selected_candidate"].eq(ADDED_CANDIDATE_ID)
        by_well_rows.append(
            {
                "well": well,
                "exp486_top1_fraction": float(added.mean()),
                "delta_fixed13_minus_parent": -0.1,
            }
        )
    by_well = pd.DataFrame(by_well_rows)
    novelty = build_postfreeze_addone_novelty_readout(
        new_score_path=new_path,
        output_dir=tmp_path / "diag",
    )
    reranking = build_incumbent_reranking_diagnostic(
        new_score_path=new_path,
        paired=paired,
        by_well=by_well,
        contract=contract(),
        output_dir=tmp_path / "diag",
        quantile_bins=4,
    )
    assert novelty["affects_training_or_scientific_gate"] is False
    assert novelty["h512"]["oracle_improvement_ft"] > 0.0
    assert reranking["affects_training_or_scientific_gate"] is False
    assert reranking["pooled_incumbent_change_fraction_when_exp486_not_top1"] == pytest.approx(1.0)
    assert (tmp_path / "diag" / "exp496_incumbent_reranking_diagnostic.csv").exists()


def test_decompressed_sha_is_independent_of_gzip_metadata(tmp_path: Path) -> None:
    payload = b"a,b\n1,2\n"
    first = tmp_path / "first.csv.gz"
    second = tmp_path / "second.csv.gz"
    with gzip.GzipFile(first, "wb", mtime=1) as stream:
        stream.write(payload)
    with gzip.GzipFile(second, "wb", mtime=2) as stream:
        stream.write(payload)
    assert first.read_bytes() != second.read_bytes()
    assert sha256_csv_payload(first) == sha256_csv_payload(second)
