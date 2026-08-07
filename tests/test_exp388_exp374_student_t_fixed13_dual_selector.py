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
from src.exp374_fixed13_candidate_cache import (
    ADDED_CANDIDATE_ID,
    BASE_CANDIDATE_IDS,
    BASE_FIXED_IDS,
    BASE_PRIMARY_IDS,
    Exp374Fixed13CandidateCache,
    base_exp264_contract,
    build_postfreeze_addone_novelty_readout,
    build_fixed13_integration_readout,
    exp374_prediction_content_sha256,
    load_exp374_predictions,
    resolve_file_by_sha,
    sha256_decompressed_gzip,
    validate_fixed13_contract,
)

ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = (
    ROOT / "experiments" / "exp388_exp374_fixed13_dual_selector_on_exp264"
)


def contract() -> dict:
    return read_yaml(EXP_DIR / "candidate_contract.yaml")


def test_exp374_input_and_execution_contract_is_frozen() -> None:
    config = read_yaml(EXP_DIR / "config.yaml")
    source = config["data"]["exp374_predictions"]
    assert source["filename"] == (
        "exp374_exp209_student_t_exact_hmm_emission_predictions.csv.gz"
    )
    assert source["allowlist_columns"] == [
        "id",
        "well_id",
        "row_idx",
        "student_t_df4_on_exp209_absolute_tvt_hmm_tvt",
        "student_t_df4_on_exp209_absolute_tvt_hmm_std",
        "student_t_df4_on_exp209_absolute_tvt_hmm_loglik",
    ]
    assert source["expected_file_sha256"] == (
        "ea6f95334b2d75ab8c96f705c453f66b856397fd222770cd15f3ae0b7fef221e"
    )
    assert source["expected_decompressed_sha256"] == (
        "668fe87da902955acee742c72d30724abb53f32050bb5d0a5c1b3dee0cbd626e"
    )
    assert source["expected_prediction_logical_sha256"] == (
        "668fe87da902955acee742c72d30724abb53f32050bb5d0a5c1b3dee0cbd626e"
    )
    assert source["candidate_requires_oof_fold"] is False
    assert config["experiment"]["route"] == "ensemble"
    assert config["experiment"]["status"] == (
        "completed_fixed13_selector_scientific_gate_failed_closed"
    )
    assert config["execution"]["run_approved"] is True
    assert config["execution"]["approval_consumed"] is True
    assert config["execution"]["trained_cpu_boosters"] == 40
    assert config["execution"]["planned_cpu_boosters"] == 40
    assert config["execution"]["parent_control_retraining"] is False
    assert (
        config["execution"]["decision"]
        == "FAIL_CLOSE_FIXED13_SELECTOR_BRANCH"
    )
    assert config["execution"]["scientific_gate_passed"] is False


def test_fixed13_contract_is_single_add_one_and_compact77() -> None:
    value = contract()
    validate_fixed13_contract(value)
    assert [item["id"] for item in value["score_candidates"]] == [
        *BASE_CANDIDATE_IDS,
        ADDED_CANDIDATE_ID,
    ]
    assert tuple(
        value["legal_domains"]["primitive_pair_bank"]["candidates"]
    ) == (*BASE_PRIMARY_IDS, ADDED_CANDIDATE_ID)
    assert tuple(
        value["legal_domains"]["primitive_fixed_bank"]["candidates"]
    ) == BASE_FIXED_IDS
    assert len(compact_feature_names(value)) == 77


def test_base_contract_remains_valid_exp264_fixed12() -> None:
    base = base_exp264_contract(contract())
    validate_candidate_contract(base)
    assert [item["id"] for item in base["score_candidates"]] == list(
        BASE_CANDIDATE_IDS
    )


def test_exp374_loader_uses_allowlist_and_records_decompressed_sha(
    tmp_path: Path,
) -> None:
    frame = pd.DataFrame(
        {
            "id": ["a_2", "b_5", "c_1", "d_6", "e_3"],
            "well_id": ["a", "b", "c", "d", "e"],
            "row_idx": [2, 5, 1, 6, 3],
            "student_t_df4_on_exp209_absolute_tvt_hmm_tvt": [
                10.0,
                20.0,
                30.0,
                40.0,
                50.0,
            ],
            "student_t_df4_on_exp209_absolute_tvt_hmm_std": [
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
            ],
            "student_t_df4_on_exp209_absolute_tvt_hmm_loglik": [
                -10.0,
                -20.0,
                -30.0,
                -40.0,
                -50.0,
            ],
            "tvt_true": [9.0, 18.0, 33.0, 39.0, 51.0],
        }
    )
    path = tmp_path / "exp374.csv.gz"
    frame.to_csv(path, index=False, compression="gzip")
    file_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    decompressed_sha = sha256_decompressed_gzip(path)
    loaded, manifest = load_exp374_predictions(
        path,
        expected_rows=5,
        expected_wells=5,
        expected_file_sha256=file_sha,
        expected_decompressed_sha256=decompressed_sha,
        expected_prediction_logical_sha256="a" * 64,
    )
    assert list(loaded.columns[:6]) == [
        "id",
        "well_id",
        "row_idx",
        "candidate_tvt",
        "candidate_std",
        "hmm_loglik",
    ]
    assert loaded["evaluation_rows_in_well"].eq(1).all()
    assert np.array_equal(loaded["loglik_per_row"], loaded["hmm_loglik"])
    assert manifest["truth_or_error_columns_loaded"] == 0
    assert manifest["header_columns"][-1] == "tvt_true"
    assert manifest["decompressed_sha256"] == decompressed_sha
    assert manifest["candidate_generation"] == "target_free_per_well_exact_hmm"
    assert manifest["candidate_requires_oof_fold"] is False
    assert manifest["source_fold_column"] is None
    assert manifest["native_confidence_finite_fraction"] == pytest.approx(1.0)
    assert manifest["id_key_mismatches"] == 0
    assert manifest["post_read_prediction_content_sha256"] == (
        exp374_prediction_content_sha256(loaded)
    )


def test_sha_resolver_ignores_absolute_patterns_during_root_glob(
    tmp_path: Path,
) -> None:
    source = tmp_path / "nested" / "candidate.bin"
    source.parent.mkdir()
    source.write_bytes(b"fixed13")
    expected_sha = hashlib.sha256(source.read_bytes()).hexdigest()

    resolved = resolve_file_by_sha(
        ["/kaggle/input/not-mounted/candidate.bin", "**/candidate.bin"],
        [tmp_path],
        expected_file_sha256=expected_sha,
        label="absolute-and-relative-pattern regression",
    )
    assert resolved == source


class _FakeBaseCache:
    def __init__(self, bundles: FoldBundle | dict[int, FoldBundle]):
        self.bundles = bundles

    def load_fold(self, fold: int) -> FoldBundle:
        if isinstance(self.bundles, dict):
            return self.bundles[fold]
        assert fold == 0
        return self.bundles


def test_fixed13_cache_appends_exp374_in_base_row_order() -> None:
    base_frame = pd.DataFrame(
        {
            "id": ["b_6", "a_2", "a_3"],
            "well": ["b", "a", "a"],
            "well_row_idx": [6, 2, 3],
            "outer_fold": [0, 0, 0],
            "md_since": [1.0, 2.0, 3.0],
            "last_known_tvt": [100.0, 100.0, 100.0],
        }
    )
    base_values = np.arange(36, dtype=np.float32).reshape(3, 12)
    confidence = {
        name: base_frame[KEY_COLUMNS].assign(confidence_valid=True)
        for name in BASE_CANDIDATE_IDS[:6]
    }
    base_bundle = FoldBundle(
        base=base_frame,
        values=base_values,
        available=np.ones_like(base_values, dtype=bool),
        confidence=confidence,
        candidate_ids=list(BASE_CANDIDATE_IDS),
        specs={
            str(item["id"]): dict(item)
            for item in base_exp264_contract(contract())["score_candidates"]
        },
    )
    exp374 = pd.DataFrame(
        {
            "well_id": ["a", "a", "b"],
            "row_idx": [2, 3, 6],
            "candidate_tvt": [202.0, 203.0, 206.0],
            "candidate_std": [2.0, 3.0, 6.0],
            "hmm_loglik": [-20.0, -20.0, -12.0],
            "evaluation_rows_in_well": [2, 2, 1],
            "loglik_per_row": [-10.0, -10.0, -12.0],
        }
    )
    cache = Exp374Fixed13CandidateCache.__new__(Exp374Fixed13CandidateCache)
    cache.contract = contract()
    cache.ids = [*BASE_CANDIDATE_IDS, ADDED_CANDIDATE_ID]
    cache.specs = {
        str(item["id"]): dict(item) for item in cache.contract["score_candidates"]
    }
    cache.base_cache = _FakeBaseCache(base_bundle)
    cache.exp374_manifest = {
        "candidate_requires_oof_fold": False
    }
    cache.exp374_by_key = exp374.set_index(["well_id", "row_idx"])[
        [
            "candidate_tvt",
            "candidate_std",
            "hmm_loglik",
            "evaluation_rows_in_well",
            "loglik_per_row",
        ]
    ]
    cache._selector_fold_audits = {}
    extended = cache.load_fold(0)
    assert extended.values.shape == (3, 13)
    assert extended.values[:, -1].tolist() == [206.0, 202.0, 203.0]
    assert cache._selector_fold_audits[0]["candidate_source_fold"] is None
    added_confidence = extended.confidence[ADDED_CANDIDATE_ID]
    assert added_confidence["confidence_valid"].all()
    assert added_confidence["sigma_tvt"].tolist() == [6.0, 2.0, 3.0]
    assert added_confidence["source_loglik"].tolist() == [-12.0, -20.0, -20.0]
    assert added_confidence["loglik_per_row"].tolist() == [-12.0, -10.0, -10.0]


def test_fixed13_cache_audits_target_free_global_key_repartition() -> None:
    bundles: dict[int, FoldBundle] = {}
    exp374_rows = []
    base_specs = {
        str(item["id"]): dict(item)
        for item in base_exp264_contract(contract())["score_candidates"]
    }
    for fold in range(5):
        well = f"selector_{fold}"
        row = 100 + fold
        base = pd.DataFrame(
            {
                "id": [f"{well}_{row}"],
                "well": [well],
                "well_row_idx": [row],
                "outer_fold": [fold],
                "md_since": [float(fold)],
                "last_known_tvt": [100.0],
            }
        )
        bundles[fold] = FoldBundle(
            base=base,
            values=np.zeros((1, 12), dtype=np.float32),
            available=np.ones((1, 12), dtype=bool),
            confidence={
                name: base[KEY_COLUMNS].assign(confidence_valid=True)
                for name in BASE_CANDIDATE_IDS[:6]
            },
            candidate_ids=list(BASE_CANDIDATE_IDS),
            specs=base_specs,
        )
        exp374_rows.append(
            {
                "well_id": well,
                "row_idx": row,
                "candidate_tvt": 200.0 + fold,
                "candidate_std": 1.0 + fold,
                "hmm_loglik": -10.0 - fold,
                "evaluation_rows_in_well": 1,
                "loglik_per_row": -10.0 - fold,
            }
        )
    exp374 = pd.DataFrame(exp374_rows)
    cache = Exp374Fixed13CandidateCache.__new__(Exp374Fixed13CandidateCache)
    cache.contract = contract()
    cache.ids = [*BASE_CANDIDATE_IDS, ADDED_CANDIDATE_ID]
    cache.specs = {
        str(item["id"]): dict(item) for item in cache.contract["score_candidates"]
    }
    cache.base_cache = _FakeBaseCache(bundles)
    cache.exp374_manifest = {"candidate_requires_oof_fold": False}
    cache.exp374_by_key = exp374.set_index(["well_id", "row_idx"])[
        [
            "candidate_tvt",
            "candidate_std",
            "hmm_loglik",
            "evaluation_rows_in_well",
            "loglik_per_row",
        ]
    ]
    cache._selector_fold_audits = {}

    for fold in range(5):
        extended = cache.load_fold(fold)
        assert extended.values[0, -1] == pytest.approx(200.0 + fold)
    audit = cache.selector_repartition_manifest(expected_rows=5)
    assert audit["passed"] is True
    assert audit["candidate_requires_oof_fold"] is False
    assert audit["candidate_source_fold"] is None
    assert audit["source_fold_equals_selector_fold_required"] is False
    assert audit["source_fold_used_as_model_feature"] is False
    assert audit["selector_fold_rows"] == {str(fold): 1 for fold in range(5)}
    assert all(
        row["candidate_source_fold"] is None
        for row in audit["overlap_by_selector_fold"]
    )


def _score_frame(
    *,
    candidate_ids: tuple[str, ...],
    primary_ids: tuple[str, ...],
    added_is_best: bool,
) -> pd.DataFrame:
    rows = []
    base_rows = [
        (f"r{2 * fold}", f"w{fold}", fold, 100.0)
        for fold in range(5)
    ] + [
        (f"r{2 * fold + 1}", f"w{fold}", fold, 1100.0)
        for fold in range(5)
    ]
    for row_id, well, fold, md_since in base_rows:
        for position, candidate in enumerate(candidate_ids):
            actual = 1.0 if candidate == "exp226_k16" else 2.0 + position / 10.0
            predicted = 1.0 + position
            if added_is_best and candidate == ADDED_CANDIDATE_ID:
                actual = 0.5
                predicted = 0.1
            if candidate == "exp226_w500_50_50":
                actual = 3.0
            rows.append(
                {
                    "id": row_id,
                    "well": well,
                    "well_row_idx": int(row_id[-1]),
                    "outer_fold": fold,
                    "md_since": md_since,
                    "candidate_id": candidate,
                    "candidate_tvt": 100.0 + position,
                    "candidate_available": True,
                    "confidence_valid": True,
                    "actual_abs_error": actual,
                    "actual_within10": 1,
                    "pred_abs_error": predicted,
                    "p_within10": 0.9,
                }
            )
    frame = pd.DataFrame(rows)
    assert set(primary_ids).issubset(set(candidate_ids))
    return frame


def test_fixed13_readout_pairs_parent_and_reports_candidate_usage(
    tmp_path: Path,
) -> None:
    new_score = _score_frame(
        candidate_ids=(*BASE_CANDIDATE_IDS, ADDED_CANDIDATE_ID),
        primary_ids=(*BASE_PRIMARY_IDS, ADDED_CANDIDATE_ID),
        added_is_best=True,
    )
    parent_score = _score_frame(
        candidate_ids=BASE_CANDIDATE_IDS,
        primary_ids=BASE_PRIMARY_IDS,
        added_is_best=False,
    )
    new_path = tmp_path / "new.parquet"
    parent_path = tmp_path / "parent.parquet"
    new_score.to_parquet(new_path, index=False)
    parent_score.to_parquet(parent_path, index=False)
    assignment = pd.DataFrame(
        {
            "well_id": [f"w{fold}" for fold in range(5)],
            "verification_like_spatial_role": ["valid"] * 5,
            "verification_like_typewell_purged_role": ["valid"] * 5,
        }
    )
    assignment_path = tmp_path / "assignment.csv"
    assignment.to_csv(assignment_path, index=False)
    guard = build_fixed13_integration_readout(
        new_score_path=new_path,
        parent_score_path=parent_path,
        hidden_like_assignment_path=assignment_path,
        contract=contract(),
        score_summary={"score_guard": {"passed": True}},
        guard_config={
            "minimum_added_candidate_primary_top1_fraction": 0.005,
            "minimum_positive_usage_folds": 4,
            "maximum_pooled_delta_rmse_vs_parent_fixed12_selector": 0.0,
            "minimum_improved_folds_vs_parent_fixed12_selector": 4,
            "maximum_near_0_250_delta_rmse": 0.02,
            "maximum_1000_plus_delta_rmse": 0.02,
            "maximum_hidden_like_delta_rmse": 0.02,
            "maximum_by_well_p95_delta_rmse": 0.25,
            "maximum_worst_well_delta_rmse": 0.25,
        },
        output_dir=tmp_path / "out",
    )
    assert guard["passed"] is True
    assert guard["student_t_usage_pooled"] == pytest.approx(1.0)
    assert guard["fixed_fallback_error_parity_max_abs_ft"] == pytest.approx(0.0)
    assert (tmp_path / "out" / "exp388_scientific_gate.json").exists()


def test_postfreeze_addone_novelty_is_diagnostic_only(tmp_path: Path) -> None:
    score = _score_frame(
        candidate_ids=(*BASE_CANDIDATE_IDS, ADDED_CANDIDATE_ID),
        primary_ids=(*BASE_PRIMARY_IDS, ADDED_CANDIDATE_ID),
        added_is_best=True,
    )
    score_path = tmp_path / "score.parquet"
    score.to_parquet(score_path, index=False)
    result = build_postfreeze_addone_novelty_readout(
        new_score_path=score_path,
        output_dir=tmp_path / "novelty",
    )
    assert result["affects_training_or_scientific_gate"] is False
    assert result["h512"]["oracle_improvement_ft"] > 0.0
    assert result["whole_well"]["strict_unique_best_fraction"] == pytest.approx(1.0)
    assert (
        tmp_path / "novelty" / "exp388_postfreeze_addone_novelty.csv"
    ).exists()


def test_decompressed_sha_is_independent_of_gzip_metadata(tmp_path: Path) -> None:
    payload = b"a,b\n1,2\n"
    first = tmp_path / "first.csv.gz"
    second = tmp_path / "second.csv.gz"
    with gzip.GzipFile(first, "wb", mtime=1) as stream:
        stream.write(payload)
    with gzip.GzipFile(second, "wb", mtime=2) as stream:
        stream.write(payload)
    assert first.read_bytes() != second.read_bytes()
    assert sha256_decompressed_gzip(first) == sha256_decompressed_gzip(second)
