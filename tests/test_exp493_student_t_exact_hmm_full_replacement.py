from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.candidate_selector_pipeline import (
    KEY_COLUMNS,
    FoldBundle,
    candidate_ids,
    compact_feature_names,
    contract_by_id,
    read_yaml,
)
from src.exact_hmm_full_replacement import (
    CHANGED_CANDIDATES,
    Exp374Fixed12ReplacementCache,
    SEMANTIC_SLOT,
    STUDENT_T_REPLACEMENT_VALUE_SOURCE,
    UNCHANGED_CANDIDATES,
    build_candidate_bank_from_primitives,
    build_fixed12_replacement_readout,
    load_student_t_replacement_predictions,
    replacement_cost_contract,
    run_fixed12_stage_a_rebuild,
    validate_fixed12_replacement_contract,
)
from src.exp374_fixed13_candidate_cache import (
    BASE_CANDIDATE_IDS,
    BASE_FIXED_IDS,
    BASE_PRIMARY_IDS,
    load_exp374_predictions,
    sha256_decompressed_gzip,
)

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments/exp493_student_t_exact_hmm_full_replacement_on_exp264"
PARENT = (
    ROOT / "experiments/exp264_exp263_candidate_confidence_dual_selector"
)


def load_contract() -> dict[str, Any]:
    return read_yaml(EXP / "candidate_contract.yaml")


def load_config() -> dict[str, Any]:
    return read_yaml(EXP / "config.yaml")


def test_exp493_cost_is_frozen_and_kaggle_run_is_approved() -> None:
    config = load_config()
    assert config["experiment"]["route"] == "ensemble"
    assert config["experiment"]["status"] == (
        "stage_c_completed_scientific_gate_failed_closed"
    )
    assert config["authorization"]["implementation_approved"] is True
    assert (
        config["authorization"]["implementation_approval_source"]
        == "user_message_implement_exp493_2026_07_31"
    )
    assert config["authorization"]["canonical_notebook_adoption_approved"] is True
    assert config["authorization"]["kaggle_package_approved"] is True
    assert config["authorization"]["kaggle_run_approved"] is True
    assert config["authorization"]["kaggle_v3_rerun_approved"] is True
    assert (
        config["authorization"]["kaggle_v3_rerun_approval_source"]
        == "user_message_rerun_exp493_2026_07_31"
    )
    assert config["implementation"] == {
        "enabled": True,
        "scaffold_created": True,
        "executable_code_created": True,
        "canonical_notebooks_placeholder_only": False,
        "jupytext_source_created": True,
        "helper_code_created": True,
        "tests_created": True,
        "kaggle_package_created": True,
    }
    assert replacement_cost_contract(config) == {
        "active_variants": 1,
        "objectives": 2,
        "outer_folds": 5,
        "inner_folds": 4,
        "planned_cpu_selector_boosters": 40,
        "parent_control_retraining": False,
        "gpu_boosters": 0,
        "downstream_tvt_training": False,
        "inference": False,
        "submission": False,
    }
    assert config["execution"]["run_approved"] is True
    assert config["execution"]["stage"] == (
        "stage_c_completed_scientific_gate_failed_closed"
    )
    assert config["execution"]["trained_cpu_boosters"] == 40
    assert config["execution"]["v3_rerun_additional_cpu_selector_boosters"] == 40
    assert config["execution"]["v3_rerun_cumulative_cpu_selector_boosters"] == 80
    assert config["execution"]["v3_rerun_requires_separate_approval"] is False
    assert config["execution"]["v3_rerun_approved_scope"] == (
        "fixed12_student_t_v3_postprocess_fix_additional40_"
        "cumulative80_cpu_boosters_no_control_retraining"
    )
    assert config["execution"]["approved_scope"] == (
        "fixed12_student_t_replacement_stage_a_stage_c_"
        "1_variant_2_objectives_outer5_inner4_40_cpu_boosters_"
        "no_control_retraining"
    )
    assert config["runtime"]["kaggle"]["bootstrap_dependency_files"] == [
        {
            "source": (
                "experiments/"
                "exp264_exp263_candidate_confidence_dual_selector/config.yaml"
            ),
            "destination": "inputs/exp264_parent_config.yaml",
        }
    ]
    assert config["runtime"]["kaggle"]["run_history"][-2] == {
        "version": 2,
        "status": "error",
        "pushed_at_utc": "2026-07-30 23:30:36 UTC",
        "failed_at_elapsed_seconds": 5846.077,
        "trained_cpu_selector_boosters": 40,
        "failed_stage": "post_scientific_gate_feature_importance",
        "scientific_gate_executed": True,
        "output_artifacts_downloadable": False,
        "error": "feature_importance_gain_column_not_found",
    }
    assert config["runtime"]["kaggle"]["run_history"][-1] == {
        "version": 3,
        "status": "complete",
        "pushed_at_utc": "2026-07-31 03:49:34 UTC",
        "completed_at_utc": "2026-07-31 05:28:52 UTC",
        "notebook_elapsed_seconds": pytest.approx(5896.184329532),
        "trained_cpu_selector_boosters": 40,
        "cumulative_cpu_selector_boosters": 80,
        "decision": "FAIL_CLOSE_FIXED12_STUDENT_T_REPLACEMENT_SELECTOR",
        "scientific_gate_passed": False,
        "selected_output_downloaded": True,
        "full_output_archive_downloaded": False,
    }
    assert config["results"]["decision"] == (
        "FAIL_CLOSE_FIXED12_STUDENT_T_REPLACEMENT_SELECTOR"
    )
    assert config["results"]["scientific_gate_passed"] is False
    assert config["results"]["hard_primary_oof_rmse"] == pytest.approx(
        8.616237400142841
    )
    assert config["results"]["fold_improvements_vs_parent"] == 3
    assert config["results"]["model_count"] == 40


def test_fixed12_replacement_contract_preserves_ids_domains_and_schema() -> None:
    contract = load_contract()
    evidence = validate_fixed12_replacement_contract(contract)
    assert tuple(evidence["candidate_order"]) == BASE_CANDIDATE_IDS
    assert tuple(evidence["primary_domain"]) == BASE_PRIMARY_IDS
    assert tuple(evidence["fixed_domain"]) == BASE_FIXED_IDS
    assert tuple(evidence["changed_candidates"]) == CHANGED_CANDIDATES
    assert tuple(evidence["unchanged_candidates"]) == UNCHANGED_CANDIDATES
    assert (
        evidence["replacement_value_source"]
        == STUDENT_T_REPLACEMENT_VALUE_SOURCE
    )
    assert len(compact_feature_names(contract)) == 74

    parent_schema = json.loads(
        (
            PARENT
            / "kaggle/output/stage_c_v6/artifacts/feature_schema.json"
        ).read_text()
    )
    parent_compact = json.loads(
        (
            PARENT
            / "kaggle/output/stage_c_v6/artifacts/compact_meta_schema.json"
        ).read_text()
    )
    assert len(parent_schema["features"]) == 88
    assert parent_schema["feature_schema_sha256"] == (
        load_config()["data"]["parent_exp264"][
            "stage_a_feature_schema_logical_sha256"
        ]
    )
    assert parent_compact["features"] == compact_feature_names(contract)


def test_student_t_loader_enforces_allowlist_decompressed_and_post_read_sha(
    tmp_path: Path,
) -> None:
    frame = pd.DataFrame(
        {
            "id": ["a_0", "a_1", "b_0", "b_1"],
            "well_id": ["a", "a", "b", "b"],
            "row_idx": [0, 1, 0, 1],
            "student_t_df4_on_exp209_absolute_tvt_hmm_tvt": [
                10.0,
                11.0,
                20.0,
                21.0,
            ],
            "student_t_df4_on_exp209_absolute_tvt_hmm_std": [
                1.0,
                1.5,
                2.0,
                2.5,
            ],
            "student_t_df4_on_exp209_absolute_tvt_hmm_loglik": [
                -4.0,
                -4.0,
                -8.0,
                -8.0,
            ],
            "forbidden_true_tvt": [9.0, 12.0, 19.0, 22.0],
        }
    )
    path = tmp_path / "exp374.csv.gz"
    frame.to_csv(path, index=False, compression="gzip")
    file_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    decompressed_sha = sha256_decompressed_gzip(path)
    _, first_manifest = load_exp374_predictions(
        path,
        expected_rows=4,
        expected_wells=2,
        expected_file_sha256=file_sha,
        expected_decompressed_sha256=decompressed_sha,
        expected_prediction_logical_sha256=decompressed_sha,
    )
    loaded, manifest = load_student_t_replacement_predictions(
        path,
        expected_rows=4,
        expected_wells=2,
        expected_file_sha256=file_sha,
        expected_decompressed_sha256=decompressed_sha,
        expected_post_read_prediction_sha256=first_manifest[
            "post_read_prediction_content_sha256"
        ],
    )
    assert list(loaded.columns[:6]) == [
        "id",
        "well_id",
        "row_idx",
        "candidate_tvt",
        "candidate_std",
        "hmm_loglik",
    ]
    assert manifest["truth_or_error_columns_loaded"] == 0
    assert "forbidden_true_tvt" in manifest["header_columns"]
    assert "forbidden_true_tvt" not in manifest["loaded_columns"]
    assert manifest["post_read_prediction_content_sha_verified"] is True

    with pytest.raises(ValueError, match="post-read prediction content SHA"):
        load_student_t_replacement_predictions(
            path,
            expected_rows=4,
            expected_wells=2,
            expected_file_sha256=file_sha,
            expected_decompressed_sha256=decompressed_sha,
            expected_post_read_prediction_sha256="0" * 64,
        )


def _base_bundle(contract: dict[str, Any], fold: int) -> FoldBundle:
    rows = 3
    well = f"well_{fold}"
    base = pd.DataFrame(
        {
            "id": [f"{well}_{row}" for row in range(rows)],
            "well": [well] * rows,
            "well_row_idx": np.arange(rows, dtype=np.int32),
            "outer_fold": np.full(rows, fold, dtype=np.int8),
            "md_since": np.arange(1, rows + 1, dtype=np.float32),
            "last_known_tvt": np.full(rows, 1000.0 + 100 * fold, dtype=np.float32),
        }
    )
    ids = candidate_ids(contract)
    specs = contract_by_id(contract)
    primitive_ids = [
        name for name in ids if str(specs[name]["kind"]) == "primitive"
    ]
    primitive_values = {
        name: (
            base["last_known_tvt"].to_numpy(np.float32)
            + np.float32(index + 1)
            * base["md_since"].to_numpy(np.float32)
        )
        for index, name in enumerate(primitive_ids)
    }
    values = build_candidate_bank_from_primitives(primitive_values, contract)
    confidence: dict[str, pd.DataFrame] = {}
    for index, name in enumerate(primitive_ids):
        conf = base[KEY_COLUMNS].copy()
        conf["candidate_id"] = name
        conf["confidence_valid"] = True
        conf["sigma_tvt"] = np.float32(index + 1)
        conf["source_loglik"] = np.float32(-(index + 1))
        conf["loglik_per_row"] = np.float32(-(index + 1) / rows)
        if name == "exp226_k16":
            conf["geometry_gr_delta"] = np.float32(0.25)
        if name == "selfgr_hmm_a070":
            conf["candidate_finite_source"] = np.float32(1.0)
            conf["selfgr_quality"] = np.float32(0.8)
            conf["selfgr_peak_tvt"] = np.float32(1000.0)
            conf["score_margin"] = np.float32(0.2)
            conf["selfgr_typewell_agreement"] = np.float32(0.7)
            conf["selfgr_valid"] = np.float32(1.0)
        if name == "beam_mean":
            conf["beam_family_std"] = np.float32(3.0)
        confidence[name] = conf
    return FoldBundle(
        base=base,
        values=values,
        available=np.ones_like(values, dtype=bool),
        confidence=confidence,
        candidate_ids=ids,
        specs=specs,
    )


class _FakeBaseCache:
    def __init__(self, bundles: dict[int, FoldBundle]):
        self.bundles = bundles

    def load_fold(self, fold: int) -> FoldBundle:
        return self.bundles[int(fold)]


def _replacement_frame(
    bundles: dict[int, FoldBundle],
    contract: dict[str, Any],
) -> pd.DataFrame:
    exact_position = candidate_ids(contract).index(SEMANTIC_SLOT)
    parts: list[pd.DataFrame] = []
    for fold, bundle in bundles.items():
        frame = pd.DataFrame(
            {
                "id": bundle.base["id"].astype(str),
                "well_id": bundle.base["well"].astype(str),
                "row_idx": bundle.base["well_row_idx"].to_numpy(np.int32),
                "candidate_tvt": (
                    bundle.values[:, exact_position] + np.float32(5.0 + fold)
                ),
                "candidate_std": np.full(len(bundle.base), 2.5, dtype=np.float32),
                "hmm_loglik": np.full(
                    len(bundle.base), -12.0 - fold, dtype=np.float32
                ),
                "evaluation_rows_in_well": np.full(
                    len(bundle.base), len(bundle.base), dtype=np.int32
                ),
                "loglik_per_row": np.full(
                    len(bundle.base),
                    (-12.0 - fold) / len(bundle.base),
                    dtype=np.float32,
                ),
            }
        )
        parts.append(frame)
    return pd.concat(parts, ignore_index=True)


def test_replacement_cache_changes_four_and_preserves_eight_candidates() -> None:
    contract = load_contract()
    bundles = {fold: _base_bundle(contract, fold) for fold in range(5)}
    replacement = _replacement_frame(bundles, contract)
    cache = Exp374Fixed12ReplacementCache(
        Path("/unused"),
        contract,
        exp374_predictions=replacement,
        exp374_manifest={
            "truth_or_error_columns_loaded": 0,
            "candidate_requires_oof_fold": False,
        },
        base_cache=_FakeBaseCache(bundles),
    )
    positions = {
        name: index for index, name in enumerate(candidate_ids(contract))
    }
    for fold in range(5):
        observed = cache.load_fold(fold)
        parent = bundles[fold]
        for name in UNCHANGED_CANDIDATES:
            assert np.array_equal(
                observed.values[:, positions[name]],
                parent.values[:, positions[name]],
            )
        for name in CHANGED_CANDIDATES:
            assert not np.array_equal(
                observed.values[:, positions[name]],
                parent.values[:, positions[name]],
            )
        exact_confidence = observed.confidence[SEMANTIC_SLOT]
        assert exact_confidence["confidence_valid"].all()
        assert exact_confidence["confidence_source"].eq(
            "exp374_student_t_df4_exact_hmm_posterior"
        ).all()
        assert np.allclose(exact_confidence["sigma_tvt"], 2.5)

    manifest = cache.replacement_manifest(expected_rows=15)
    assert manifest["passed"] is True
    assert manifest["unchanged_candidate_max_abs_ft"] == 0.0
    assert manifest["formula_parity_max_abs_ft"] <= 1.0e-6
    assert manifest["checks"]["truth_or_error_columns_loaded_before_feature_freeze"]


def test_replacement_cache_fails_closed_on_missing_global_key() -> None:
    contract = load_contract()
    bundles = {fold: _base_bundle(contract, fold) for fold in range(5)}
    replacement = _replacement_frame(bundles, contract).iloc[1:].copy()
    cache = Exp374Fixed12ReplacementCache(
        Path("/unused"),
        contract,
        exp374_predictions=replacement,
        exp374_manifest={
            "truth_or_error_columns_loaded": 0,
            "candidate_requires_oof_fold": False,
        },
        base_cache=_FakeBaseCache(bundles),
    )
    with pytest.raises(ValueError, match="global key join misses"):
        cache.load_fold(0)


def test_stage_a_rebuild_reuses_exact_parent_88_and_74_without_truth(
    tmp_path: Path,
) -> None:
    contract = load_contract()
    config = load_config()
    config["validation"]["expected_rows"] = 15
    config["validation"]["expected_wells"] = 5
    parent_config = read_yaml(PARENT / "config.yaml")
    bundles = {fold: _base_bundle(contract, fold) for fold in range(5)}
    replacement = _replacement_frame(bundles, contract)
    cache = Exp374Fixed12ReplacementCache(
        Path("/unused"),
        contract,
        exp374_predictions=replacement,
        exp374_manifest={
            "truth_or_error_columns_loaded": 0,
            "candidate_requires_oof_fold": False,
            "post_read_prediction_content_sha256": "synthetic",
        },
        base_cache=_FakeBaseCache(bundles),
    )
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    for bundle in bundles.values():
        well = str(bundle.base["well"].iloc[0])
        horizontal = pd.DataFrame(
            {
                "MD": [100.0, 101.0, 102.0],
                "X": [10.0, 11.0, 12.0],
                "Y": [20.0, 21.0, 22.0],
                "Z": [30.0, 31.0, 32.0],
                "GR": [40.0, 41.0, 42.0],
                "TVT_input": [1000.0, np.nan, np.nan],
            }
        )
        typewell = pd.DataFrame(
            {
                "TVT": [900.0, 1000.0, 1100.0],
                "GR": [35.0, 40.0, 45.0],
            }
        )
        horizontal.to_csv(raw_dir / f"{well}__horizontal_well.csv", index=False)
        typewell.to_csv(raw_dir / f"{well}__typewell.csv", index=False)

    output_dir = tmp_path / "artifacts"
    summary = run_fixed12_stage_a_rebuild(
        config=config,
        parent_config=parent_config,
        contract=contract,
        cache=cache,
        raw_train_dir=raw_dir,
        parent_feature_schema_path=(
            PARENT
            / "kaggle/output/stage_c_v6/artifacts/feature_schema.json"
        ),
        parent_feature_catalog_path=(
            PARENT
            / "kaggle/output/stage_c_v6/artifacts/feature_catalog.csv"
        ),
        parent_compact_schema_path=(
            PARENT
            / "kaggle/output/stage_c_v6/artifacts/compact_meta_schema.json"
        ),
        output_dir=output_dir,
    )
    assert summary["passed"] is True
    assert summary["models_trained"] == 0
    assert summary["truth_rows_loaded_before_feature_freeze"] == 0
    assert summary["feature_count"] == 88
    assert summary["compact_feature_count"] == 74
    assert (output_dir / "feature_schema.json").read_bytes() == (
        PARENT / "kaggle/output/stage_c_v6/artifacts/feature_schema.json"
    ).read_bytes()
    assert (output_dir / "compact_meta_schema.json").read_bytes() == (
        PARENT
        / "kaggle/output/stage_c_v6/artifacts/compact_meta_schema.json"
    ).read_bytes()


def _score_frame(
    *,
    replacement: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    assignment_rows: list[dict[str, Any]] = []
    for fold in range(5):
        for distance_index, md_since in enumerate((100.0, 1200.0)):
            well = f"well_{fold}_{distance_index}"
            row_id = f"{well}_0"
            assignment_rows.append(
                {
                    "well_id": well,
                    "verification_like_spatial_role": "valid",
                    "verification_like_typewell_purged_role": "valid",
                }
            )
            selected = "exact_hmm" if replacement else "exp226_k16"
            for candidate in BASE_CANDIDATE_IDS:
                if candidate == selected:
                    predicted_error = 0.1
                elif candidate in BASE_PRIMARY_IDS:
                    predicted_error = 10.0
                else:
                    predicted_error = 20.0
                if candidate == selected:
                    actual_error = 1.0 if replacement else 2.0
                elif candidate == "exp226_w500_50_50":
                    actual_error = 2.5 if replacement else 3.0
                else:
                    actual_error = 4.0
                rows.append(
                    {
                        "id": row_id,
                        "well": well,
                        "well_row_idx": 0,
                        "outer_fold": fold,
                        "md_since": md_since,
                        "candidate_id": candidate,
                        "actual_abs_error": actual_error,
                        "pred_abs_error": predicted_error,
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(assignment_rows)


def test_replacement_readout_allows_fixed_fallback_change_and_applies_gate(
    tmp_path: Path,
) -> None:
    parent, assignment = _score_frame(replacement=False)
    candidate, _ = _score_frame(replacement=True)
    parent_path = tmp_path / "parent.parquet"
    candidate_path = tmp_path / "candidate.parquet"
    assignment_path = tmp_path / "hidden.csv"
    parent.to_parquet(parent_path, index=False)
    candidate.to_parquet(candidate_path, index=False)
    assignment.to_csv(assignment_path, index=False)

    gate = build_fixed12_replacement_readout(
        new_score_path=candidate_path,
        parent_score_path=parent_path,
        hidden_like_assignment_path=assignment_path,
        contract=load_contract(),
        score_summary={"score_guard": {"passed": True}},
        technical_checks={"all_synthetic_checks": True},
        saved_control={
            "hard_primary_oof_rmse": 2.0,
            "fixed_fallback_oof_rmse": 3.0,
        },
        guard_config={
            "maximum_pooled_delta_rmse_vs_parent_fixed12_selector": 0.0,
            "minimum_improved_folds_vs_parent_fixed12_selector": 4,
            "maximum_near_0_250_delta_rmse_ft": 0.02,
            "maximum_1000_plus_delta_rmse_ft": 0.02,
            "maximum_hidden_like_delta_rmse_ft": 0.02,
            "maximum_by_well_p95_delta_rmse_ft": 0.25,
            "maximum_worst_well_delta_rmse_ft": 0.25,
            "pass_action": "qualify_followup",
            "fail_action": "close",
        },
        output_dir=tmp_path / "out",
        artifact_prefix="exp493",
    )
    assert gate["passed"] is True
    assert gate["replacement_hard_rmse"] == pytest.approx(1.0)
    assert gate["delta_replacement_minus_parent"] == pytest.approx(-1.0)
    assert gate["fold_improvements_vs_parent"] == 5
    assert gate["changed_family_usage"]["policy"] == "report_only"
    assert gate["changed_family_usage"]["top1_fraction"] == pytest.approx(1.0)
    assert gate["fixed_fallback"]["policy"].startswith("report_only")
    assert gate["fixed_fallback"]["delta_replacement_minus_parent"] == (
        pytest.approx(-0.5)
    )


def test_compact_jupytext_source_and_adopted_canonical_are_readable() -> None:
    source_path = (
        EXP
        / "exp493_student_t_exact_hmm_full_replacement_on_exp264_compact_selfcontained_train.py"
    )
    source = source_path.read_text()
    for required in [
        "validate_fixed12_replacement_contract(CONTRACT)",
        "run_fixed12_stage_a_rebuild(",
        "run_stage_c(",
        "build_fixed12_replacement_readout(",
        "nested_feature_importance_by_objective_outer_inner.csv",
        'importance["importance_type"].eq("gain")',
        "expected_v3_rerun_approval_scope",
        "Parent/control retrained:",
        "Inference executed:",
        "Submission generated or submitted:",
    ]:
        assert required in source
    assert "__file__" not in source
    assert source.count("# %% [markdown]") >= 10
    assert source.index("run_fixed12_stage_a_rebuild(") < source.index(
        "run_stage_c("
    )
    assert source.index("run_stage_c(") < source.index(
        "build_fixed12_replacement_readout("
    )

    canonical_path = (
        EXP / "exp493_student_t_exact_hmm_full_replacement_on_exp264_train.ipynb"
    )
    canonical = json.loads(canonical_path.read_text())
    assert canonical["cells"]
    assert any(
        cell["cell_type"] == "code" and cell.get("source")
        for cell in canonical["cells"]
    )
    canonical_source = "".join(
        "".join(cell.get("source", [])) for cell in canonical["cells"]
    )
    for required in [
        "run_fixed12_stage_a_rebuild(",
        "run_stage_c(",
        "build_fixed12_replacement_readout(",
    ]:
        assert required in canonical_source
