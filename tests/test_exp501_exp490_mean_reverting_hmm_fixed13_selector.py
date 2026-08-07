from __future__ import annotations

import gzip
import hashlib
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
from src.exp490_fixed13_candidate_cache import (
    ADDED_CANDIDATE_ID,
    BASE_CANDIDATE_IDS,
    BASE_FIXED_IDS,
    BASE_PRIMARY_IDS,
    EXP490_INPUT_ALLOWLIST,
    EXP490_NATIVE_FIELDS,
    Exp490Fixed13CandidateCache,
    base_exp264_contract,
    build_fixed13_integration_readout,
    build_incumbent_reranking_diagnostic,
    build_postfreeze_addone_novelty_readout,
    exp490_content_sha256,
    load_exp490_target_free_inputs,
    pair_selector_scores,
    sha256_csv_payload,
    validate_fixed13_contract,
)

ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = (
    ROOT
    / "experiments"
    / "exp501_exp490_mean_reverting_hmm_fixed13_selector_on_exp264"
)


def contract() -> dict:
    return read_yaml(EXP_DIR / "candidate_contract.yaml")


def test_repository_config_records_completed_fail_closed_stage_a_c_run() -> None:
    config = read_yaml(EXP_DIR / "config.yaml")
    assert config["experiment"]["route"] == "ensemble"
    assert config["experiment"]["status"] == "completed_train_side_rejected_no_submit"
    assert config["implementation"]["approved"] is True
    assert config["implementation"]["code_created"] is True
    assert config["implementation"]["contracts_created"] is True
    assert config["implementation"]["canonical_notebooks_are_placeholders"] is False
    assert config["implementation"]["canonical_train_notebook_adopted"] is True
    assert config["implementation"]["canonical_inference_notebook_is_placeholder"] is True
    assert config["implementation"]["kaggle_package_created"] is True
    assert config["execution"]["run_approved"] is True
    assert config["execution"]["approved_scope"] == (
        "fixed13_stage_a_plus_stage_c_1_variant_2_objectives_"
        "5_outer_4_inner_40_cpu_boosters_no_control_retraining"
    )
    assert config["execution"]["planned_cpu_boosters"] == 40
    assert config["execution"]["trained_cpu_selector_boosters"] == 40
    assert config["execution"]["parent_control_retraining"] is False
    assert config["execution"]["candidate_hmm_rerun_well_runs"] == 0
    assert config["execution"]["candidate_pf_rerun_well_runs"] == 0
    assert config["execution"]["beam_well_runs"] == 0
    assert config["execution"]["gpu_boosters"] == 0
    assert config["execution"]["inference"] is False
    assert config["execution"]["submission"] is False
    assert config["results"]["technical_gate_passed"] is True
    assert config["results"]["leakage_gate_passed"] is True
    assert config["results"]["selector_score_guard_passed"] is True
    assert config["results"]["scientific_gate_passed"] is False
    assert config["results"]["decision"] == (
        "FAIL_CLOSE_EXP490_MEAN_REVERTING_HMM_FIXED13_SELECTOR"
    )


def test_fixed13_contract_is_single_exp490_add_one() -> None:
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
    assert tuple(value["legal_domains"]["primitive_fixed_bank"]["candidates"]) == (
        BASE_FIXED_IDS
    )
    assert len(compact_feature_names(value)) == 77
    assert value["added_candidate_contract"]["new_pair_or_blend_candidates"] == []


def test_base_contract_remains_valid_exp264_fixed12() -> None:
    base = base_exp264_contract(contract())
    validate_candidate_contract(base)
    assert [item["id"] for item in base["score_candidates"]] == list(BASE_CANDIDATE_IDS)


def _write_exp490_source(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    source = pd.DataFrame(
        {
            "id": ["a_1", "a_2", "b_3"],
            "well": ["a", "a", "b"],
            "row_idx": [1, 2, 3],
            "suffix_offset": [0, 1, 0],
            "geometry_mean_reverting_hmm": [10.0, 11.0, 20.0],
            "geometry_mean_reverting_delta_mean": [0.1, 0.2, -0.3],
            "geometry_mean_reverting_hmm_std": [1.0, 2.0, 3.0],
            "fold": [0, 0, 1],
            "true_tvt_readout_only": [9.0, 9.0, 19.0],
            "candidate_error": [1.0, 2.0, 1.0],
            "episode_id": ["x", "x", "y"],
        }
    )
    path = tmp_path / "predictions.csv.gz"
    source.to_csv(path, index=False, compression="gzip")
    return path, {
        "raw": hashlib.sha256(path.read_bytes()).hexdigest(),
        "payload": sha256_csv_payload(path),
    }


def test_exp490_loader_uses_one_strict_allowlist_and_truth_late_boundary(
    tmp_path: Path,
) -> None:
    path, sha = _write_exp490_source(tmp_path)
    loaded, manifest = load_exp490_target_free_inputs(
        path,
        expected_rows=3,
        expected_wells=2,
        expected_prediction_gzip_raw_sha256=sha["raw"],
        expected_prediction_payload_sha256=sha["payload"],
    )
    assert list(manifest["prediction_loaded_columns"]) == list(EXP490_INPUT_ALLOWLIST)
    assert list(loaded.columns) == [
        "id",
        "well_id",
        "row_idx",
        "suffix_offset",
        "candidate_tvt",
        *EXP490_NATIVE_FIELDS,
    ]
    assert "fold" in manifest["prediction_header_columns"]
    assert "true_tvt_readout_only" in manifest["prediction_header_columns"]
    assert manifest["forbidden_truth_error_role_episode_fold_scope_gate_columns_loaded"] == 0
    assert manifest["upstream_source_fold_column_loaded"] is False
    assert manifest["upstream_source_fold_used_as_model_feature"] is False
    assert manifest["candidate_and_native_confidence_finite_fraction"] == 1.0
    assert manifest["post_read_content_sha256"] == exp490_content_sha256(loaded)


def test_exp490_loader_rejects_noncontiguous_suffix_offset(tmp_path: Path) -> None:
    path, sha = _write_exp490_source(tmp_path)
    frame = pd.read_csv(path)
    frame.loc[1, "suffix_offset"] = 3
    frame.to_csv(path, index=False, compression="gzip")
    sha = {
        "raw": hashlib.sha256(path.read_bytes()).hexdigest(),
        "payload": sha256_csv_payload(path),
    }
    with pytest.raises(ValueError, match="suffix-offset sequence mismatch"):
        load_exp490_target_free_inputs(
            path,
            expected_rows=3,
            expected_wells=2,
            expected_prediction_gzip_raw_sha256=sha["raw"],
            expected_prediction_payload_sha256=sha["payload"],
        )


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
        name: base[KEY_COLUMNS].assign(confidence_valid=True)
        for name in BASE_CANDIDATE_IDS[:6]
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


def _uninitialized_cache(
    bundles: FoldBundle | dict[int, FoldBundle], source: pd.DataFrame
) -> Exp490Fixed13CandidateCache:
    cache = Exp490Fixed13CandidateCache.__new__(Exp490Fixed13CandidateCache)
    cache.contract = contract()
    cache.ids = [*BASE_CANDIDATE_IDS, ADDED_CANDIDATE_ID]
    cache.specs = {str(item["id"]): dict(item) for item in cache.contract["score_candidates"]}
    cache.base_cache = _FakeBaseCache(bundles)
    cache.exp490_manifest = {
        "upstream_exp490_group_safe_oof_contract": True,
        "upstream_source_fold_column_loaded": False,
        "upstream_source_fold_used_as_model_feature": False,
    }
    cache.exp490_by_key = source.set_index(["well_id", "row_idx"])[
        ["candidate_tvt", "suffix_offset", *EXP490_NATIVE_FIELDS]
    ]
    cache._selector_fold_audits = {}
    return cache


def test_fixed13_cache_appends_exp490_in_base_row_order_and_checks_suffix() -> None:
    base = _base_bundle(0, [("b", 6), ("a", 2), ("a", 3)])
    source = pd.DataFrame(
        {
            "well_id": ["a", "a", "b"],
            "row_idx": [2, 3, 6],
            "candidate_tvt": [202.0, 203.0, 206.0],
            "suffix_offset": [0, 1, 0],
            "geometry_mean_reverting_delta_mean": [2.0, 3.0, 6.0],
            "geometry_mean_reverting_hmm_std": [2.0, 3.0, 6.0],
        }
    )
    cache = _uninitialized_cache(base, source)
    extended = cache.load_fold(0)
    assert extended.values[:, -1].tolist() == [206.0, 202.0, 203.0]
    added_confidence = extended.confidence[ADDED_CANDIDATE_ID]
    assert added_confidence["confidence_valid"].all()
    assert added_confidence["geometry_mean_reverting_delta_mean"].tolist() == [
        6.0,
        2.0,
        3.0,
    ]

    bad = source.copy()
    bad.loc[1, "suffix_offset"] = 2
    with pytest.raises(ValueError, match="suffix-offset parity failed"):
        _uninitialized_cache(base, bad).load_fold(0)


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
                "geometry_mean_reverting_delta_mean": 0.1,
                "geometry_mean_reverting_hmm_std": 1.0,
            }
        )
    cache = _uninitialized_cache(bundles, pd.DataFrame(rows))
    for fold in range(5):
        cache.load_fold(fold)
    audit = cache.selector_repartition_manifest(expected_rows=5)
    assert audit["passed"] is True
    assert audit["checks"]["suffix_offset_parity"] is True
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
        pd.DataFrame({"GR": [10.0, np.nan]}).to_csv(
            raw_dir / f"w{fold}__horizontal_well.csv", index=False
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


def test_integration_and_postfreeze_readouts_use_exp501_contract(tmp_path: Path) -> None:
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
        "failure_decision": "FAIL_CLOSE_EXP490_MEAN_REVERTING_HMM_FIXED13_SELECTOR",
    }
    output_dir = tmp_path / "out"
    gate, by_well = build_fixed13_integration_readout(
        paired=paired,
        hidden_like_assignment_path=assignment_path,
        raw_train_dir=raw_dir,
        score_summary={"score_guard": {"passed": True}},
        guard_config=guard_config,
        output_dir=output_dir,
    )
    assert gate["passed"] is True
    assert gate["positive_exp490_usage_folds"] == 5
    assert by_well["exp490_top1_fraction"].eq(0.5).all()
    assert (output_dir / "exp501_scientific_gate.json").exists()

    novelty = build_postfreeze_addone_novelty_readout(
        new_score_path=new_path,
        output_dir=output_dir,
    )
    reranking = build_incumbent_reranking_diagnostic(
        new_score_path=new_path,
        paired=paired,
        by_well=by_well,
        contract=contract(),
        output_dir=output_dir,
        quantile_bins=4,
    )
    assert novelty["candidate"] == ADDED_CANDIDATE_ID
    assert novelty["affects_training_or_scientific_gate"] is False
    assert reranking["affects_training_or_scientific_gate"] is False
    assert reranking[
        "pooled_incumbent_change_fraction_when_exp490_not_top1"
    ] == pytest.approx(1.0)
    assert (output_dir / "exp501_postfreeze_addone_novelty.json").exists()
    assert (output_dir / "exp501_incumbent_reranking_summary.json").exists()


def test_notebook_candidates_have_full_train_sections_and_fail_closed_inference() -> None:
    train_name = (
        "exp501_exp490_mean_reverting_hmm_fixed13_selector_on_exp264_"
        "compact_selfcontained_train.py"
    )
    inference_name = (
        "exp501_exp490_mean_reverting_hmm_fixed13_selector_on_exp264_"
        "compact_selfcontained_inference.py"
    )
    train = (EXP_DIR / train_name).read_text()
    inference = (EXP_DIR / inference_name).read_text()
    assert train.count("# %% [markdown]") >= 11
    for chapter in range(1, 10):
        assert f"# ## {chapter}." in train
    assert "Path(__file__)" not in train
    assert "EXP501_IMPORT_ONLY" in train
    assert "run_approved" in train
    assert "40 CPU boosters" in train
    assert "raise RuntimeError" in inference
    assert "submission" in inference


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
