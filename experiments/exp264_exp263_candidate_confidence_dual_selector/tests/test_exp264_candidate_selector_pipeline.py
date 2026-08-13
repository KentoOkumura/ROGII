from __future__ import annotations

import ast
import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.candidate_selector_pipeline import (
    FoldBundle,
    ShapeState,
    add_candidate_labels,
    apply_stage_d_base_feature_allowlist,
    audit_feature_frame,
    audit_raw_context_availability,
    build_candidate_long_features,
    build_compact_meta,
    build_nested_inner_fold_maps,
    candidate_ids,
    compact_feature_names,
    contract_by_id,
    current_test_bundle_from_wide,
    resolve_stage_c_artifact_root,
    sha256_file,
    stage_d_cost_contract,
    stage_d_matched_guard,
    validate_candidate_contract,
    validate_current_test_native_confidence,
    validate_inference_feature_missingness,
    verify_stage_c_artifact_root,
)
from src.fold_safe_formation_pipeline import (
    canonical_formation_feature_names,
    load_formation_feature_contract,
)
from tests.test_support import require_saved_files

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments" / "exp264_exp263_candidate_confidence_dual_selector"


def load_contract() -> dict:
    return yaml.safe_load((EXP / "candidate_contract.yaml").read_text())


def load_feature_config(contract: dict) -> dict:
    config = yaml.safe_load((EXP / "config.yaml").read_text())
    feature_config = dict(config["features"])
    feature_config["primary_domain"] = contract["legal_domains"]["primitive_pair_bank"][
        "candidates"
    ]
    feature_config["fixed_domain"] = contract["legal_domains"]["primitive_fixed_bank"]["candidates"]
    return feature_config


def test_fixed_audit_contract_selects_exactly_the_canonical_74() -> None:
    path = EXP / "assets" / "formation_74_contract.csv"
    expected_digest = "2ed254aac8e81fc3329fe25f3b34c7d8e5ab81bdac2d3f2c3901621ac3203eb4"

    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_digest
    features, evidence = load_formation_feature_contract(
        path, expected_sha256=expected_digest
    )
    assert features == canonical_formation_feature_names()
    assert evidence["selected_rows"] == 74


def test_oof_selector_probe_aligns_stage_c_by_unique_id_not_row_offset() -> None:
    source = (
        EXP
        / "exp264_exp263_candidate_confidence_dual_selector_oof_selector_confidence_probe.py"
    ).read_text()
    assert "base_id_index = pd.Index(base_ids)" in source
    assert "row_positions = base_id_index.get_indexer(ids[:, 0])" in source
    assert "covered_rows[row_positions] = True" in source
    stage_c_surface = source.split("score_file = pq.ParquetFile", maxsplit=1)[1]
    assert "base_ids[offset:stop]" not in stage_c_surface


def test_oof_selector_probe_preserves_exp238_plot_contract() -> None:
    source_path = (
        EXP
        / "exp264_exp263_candidate_confidence_dual_selector_oof_selector_confidence_probe.py"
    )
    source = source_path.read_text()
    tree = ast.parse(source)
    assignments = {
        node.targets[0].id: node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    }
    assert ast.literal_eval(assignments["REFERENCE_LINE_COLORS"]) == {
        "true_tvt": "black",
        "ml_oof": "#e11d48",
        "selector_top1": "#64748b",
        "pf_ancc": "#1f77b4",
        "beam_mean": "#ff7f0e",
        "likpf_mean": "#2ca02c",
        "exp226_k16": "#a16207",
        "exp209_hmm": "#7c3aed",
        "exp209_hmm_band": "#8b5cf6",
        "z_likpf_minmax": "#db2777",
        "confidence_margin": "#0f172a",
        "grid": "#e2e8f0",
        "caveat": "#7f1d1d",
    }

    plot_function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "plot_one_well"
    )
    plot_source = ast.unparse(plot_function)
    assert "'height_ratios': [7.0, 1.5, 0.65]" in plot_source
    assert "hmm_mean - 2.0 * hmm_std" in plot_source
    assert "hmm_mean + 2.0 * hmm_std" in plot_source
    assert "'exact HMM ±2σ'" in plot_source
    assert "minmax_negative_z_to_likpf(group)" in plot_source
    assert "ax_probability" not in plot_source
    assert "selector_fixed_error_margin" not in plot_source
    assert "selector_primary_probability_margin" in plot_source  # summary-only metric

    main_plot_calls = [
        ast.unparse(node)
        for node in ast.walk(plot_function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "plot"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "ax_tvt"
    ]
    joined_main_calls = "\n".join(main_plot_calls)
    for required in [
        "truth",
        "final_oof",
        "primary_tvt",
        "group['likpf_mean']",
        "group['pf_ancc']",
        "group['beam_mean']",
        "group['exp226_k16_tvt']",
        "hmm_mean",
        "minmax_negative_z_to_likpf(group)",
    ]:
        assert required in joined_main_calls
    assert "full_control" not in joined_main_calls
    assert "full_fixed" not in joined_main_calls


def synthetic_bundle(contract: dict) -> FoldBundle:
    ids = candidate_ids(contract)
    specs = contract_by_id(contract)
    rows_per_well = 8
    wells = np.repeat(["well_a", "well_b"], rows_per_well)
    row_idx = np.tile(np.arange(10, 10 + rows_per_well), 2)
    base = pd.DataFrame(
        {
            "id": [f"{well}_{row}" for well, row in zip(wells, row_idx, strict=True)],
            "well": wells,
            "well_row_idx": row_idx.astype(np.int32),
            "outer_fold": np.repeat(np.int8(0), len(wells)),
            "md_since": np.tile(np.arange(1, rows_per_well + 1), 2).astype(np.float32),
            "last_known_tvt": np.repeat([1000.0, 2000.0], rows_per_well).astype(np.float32),
        }
    )
    primitive = [name for name in ids if specs[name]["kind"] == "primitive"]
    values_by_id = {
        name: (
            base["last_known_tvt"].to_numpy(np.float32)
            + np.float32(position + 1) * base["md_since"].to_numpy(np.float32)
        )
        for position, name in enumerate(primitive)
    }
    for name in ids:
        if name in values_by_id:
            continue
        parents = specs[name]["parents"]
        weights = np.asarray(specs[name]["weights"], dtype=np.float32)
        values_by_id[name] = np.column_stack([values_by_id[parent] for parent in parents]) @ weights
    values = np.column_stack([values_by_id[name] for name in ids]).astype(np.float32)
    confidence = {}
    for position, name in enumerate(primitive):
        frame = base[["id", "well", "well_row_idx", "outer_fold", "md_since"]].copy()
        frame["candidate_id"] = name
        frame["confidence_valid"] = position % 2 == 0
        frame["sigma_tvt"] = (
            np.float32(position + 1) + base["md_since"].to_numpy(np.float32) / 100.0
        )
        frame["loglik_per_row"] = -frame["sigma_tvt"]
        confidence[name] = frame
    return FoldBundle(
        base=base,
        values=values,
        available=np.ones_like(values, dtype=bool),
        confidence=confidence,
        candidate_ids=ids,
        specs=specs,
    )


def synthetic_context(bundle: FoldBundle) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ctx__md_since": bundle.base["md_since"].to_numpy(np.float32),
            "ctx__last_known_tvt": bundle.base["last_known_tvt"].to_numpy(np.float32),
            "ctx__eval_len": np.float32(8),
            "ctx__evaluation_progress": np.tile(np.arange(1, 9, dtype=np.float32) / 8.0, 2),
        }
    )


def test_candidate_long_contract_and_label_isolation() -> None:
    contract = load_contract()
    validate_candidate_contract(contract)
    excluded = contract["not_in_current_exp264_score_bank"]
    assert "train_only_primitives" not in excluded
    assert len(excluded["stage0_oof_only_not_in_current_stage1_primitives"]) == 6
    bundle = synthetic_bundle(contract)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", pd.errors.PerformanceWarning)
        features, metadata = build_candidate_long_features(
            bundle,
            synthetic_context(bundle),
            np.arange(len(bundle.base)),
            load_feature_config(contract),
            shape_state=ShapeState.from_bundle(bundle.base, bundle.values),
        )
    assert not [item for item in caught if issubclass(item.category, pd.errors.PerformanceWarning)]
    assert len(features) == len(bundle.base) * 12
    assert not any("candidate_index" in column for column in features)
    assert not {"true_tvt", "candidate_abs_error", "candidate_within10"}.intersection(features)
    id_columns = [column for column in features if column.startswith("id__candidate__")]
    assert len(id_columns) == 12
    assert np.array_equal(features[id_columns].sum(axis=1).to_numpy(), np.ones(len(features)))
    assert features["cand__tvt"].notna().all()
    assert features["bank__candidate_mean_abs_disagreement"].notna().all()

    truth = bundle.base["last_known_tvt"].to_numpy(np.float32) + 3.0
    labels = add_candidate_labels(metadata, truth, 12)
    assert labels["candidate_abs_error"].notna().all()
    assert "candidate_abs_error" not in features


def test_compact_meta_uses_only_two_legal_domains() -> None:
    contract = load_contract()
    bundle = synthetic_bundle(contract)
    rng = np.random.default_rng(42)
    expected_error = rng.uniform(0, 20, size=bundle.values.shape).astype(np.float32)
    probability = rng.uniform(0.01, 0.99, size=bundle.values.shape).astype(np.float32)
    compact = build_compact_meta(
        bundle.base,
        bundle.values,
        expected_error,
        probability,
        bundle.available,
        np.ones_like(bundle.available, dtype=bool),
        contract,
    )
    selector_columns = [column for column in compact if column.startswith("selector__")]
    assert selector_columns == compact_feature_names(contract)
    assert not any("all_12" in column for column in selector_columns)
    assert any("primitive_pair_bank" in column for column in selector_columns)
    assert any("primitive_fixed_bank" in column for column in selector_columns)
    assert "selector__fixed_top1_is_fixed" in selector_columns


def test_feature_audit_drops_only_exact_duplicate_and_constant() -> None:
    frame = pd.DataFrame(
        {
            "ctx__x": np.arange(100, dtype=np.float32),
            "ctx__x_copy": np.arange(100, dtype=np.float32),
            "ctx__constant": np.ones(100, dtype=np.float32),
            "ctx__near": np.arange(100, dtype=np.float32) + 1e-4,
        }
    )
    config = {
        "audit": {
            "correlation_long_rows": 100,
            "pearson_abs_threshold": 0.999,
            "spearman_abs_threshold": 0.999,
        }
    }
    catalog, selected, correlation = audit_feature_frame(frame, config)
    assert "ctx__x" in selected
    assert "ctx__x_copy" not in selected
    assert "ctx__constant" not in selected
    assert "ctx__near" in selected
    assert (correlation["relation"] == "exact_duplicate").any()
    assert (correlation["relation"] == "near_duplicate_report_only").any()
    assert catalog.set_index("feature").loc["ctx__near", "selected"]


def test_raw_context_availability_is_checked_against_actual_split_schemas(
    tmp_path: Path,
) -> None:
    train_dir = tmp_path / "train"
    test_dir = tmp_path / "test"
    train_dir.mkdir()
    test_dir.mkdir()
    pd.DataFrame(columns=["MD", "X", "Y", "Z", "GR", "ANCC", "TVT_input"]).to_csv(
        train_dir / "train__horizontal_well.csv", index=False
    )
    pd.DataFrame(columns=["MD", "X", "Y", "Z", "GR", "TVT_input"]).to_csv(
        test_dir / "test__horizontal_well.csv", index=False
    )

    audit = audit_raw_context_availability(
        train_dir, test_dir, ["MD", "X", "Y", "Z", "GR"]
    )
    assert audit["availability_pass"].all()

    with pytest.raises(ValueError, match="ANCC"):
        audit_raw_context_availability(
            train_dir, test_dir, ["MD", "X", "Y", "Z", "ANCC", "GR"]
        )


def test_current_test_wide_adapter_preserves_candidate_order() -> None:
    contract = load_contract()
    bundle = synthetic_bundle(contract)
    wide = bundle.base[["id", "well", "well_row_idx"]].copy()
    for position, name in enumerate(bundle.candidate_ids):
        wide[name] = bundle.values[:, position]
    wide["confidence__exact_hmm__sigma_tvt"] = np.float32(2.5)
    wide["confidence__exact_hmm__confidence_valid"] = True
    adapted = current_test_bundle_from_wide(wide, contract)
    assert adapted.candidate_ids == bundle.candidate_ids
    assert np.array_equal(adapted.values, bundle.values)
    assert adapted.base["outer_fold"].eq(-1).all()
    assert adapted.confidence["exact_hmm"]["confidence_valid"].all()


def test_current_test_native_confidence_contract_is_fail_closed() -> None:
    contract = load_contract()
    bundle = synthetic_bundle(contract)
    wide = bundle.base[["id", "well", "well_row_idx"]].copy()
    for position, name in enumerate(bundle.candidate_ids):
        wide[name] = bundle.values[:, position]
    fields_by_primitive = contract["confidence_contract"][
        "current_test_required_fields_by_primitive"
    ]
    for candidate_position, (candidate_id, fields) in enumerate(
        fields_by_primitive.items()
    ):
        for field_position, field in enumerate(fields):
            column = f"confidence__{candidate_id}__{field}"
            if field == "confidence_valid":
                wide[column] = candidate_id != "likpf_mean"
            else:
                wide[column] = np.float32(candidate_position + field_position / 10.0)

    summary = validate_current_test_native_confidence(wide, contract)
    assert summary["required_column_count"] == 21
    assert summary["coverage"]["likpf_mean"]["valid_rate"] == 0.0
    assert summary["coverage"]["selfgr_hmm_a070"]["valid_rate"] == 1.0

    with pytest.raises(ValueError, match="beam_family_std"):
        validate_current_test_native_confidence(
            wide.drop(columns="confidence__beam_mean__beam_family_std"), contract
        )


def test_inference_missingness_guard_preserves_structural_nan_semantics() -> None:
    frame = pd.DataFrame(
        {
            "ctx__dense": [1.0, 2.0, 3.0, 4.0],
            "conf__native__sigma_tvt": [1.0, np.nan, 2.0, np.nan],
            "formula__component_std": [np.nan, 0.5, np.nan, 0.75],
        }
    )
    training_missing_rates = {
        "ctx__dense": 0.0,
        "conf__native__sigma_tvt": 0.5,
        "formula__component_std": 0.5,
    }
    summary = validate_inference_feature_missingness(frame, training_missing_rates)
    assert summary.set_index("feature").loc[
        "conf__native__sigma_tvt", "missing_count"
    ] == 2
    assert summary["structural_missingness"].tolist() == [False, True, True]

    dense_nan = frame.copy()
    dense_nan.loc[0, "ctx__dense"] = np.nan
    with pytest.raises(ValueError, match="training-dense"):
        validate_inference_feature_missingness(dense_nan, training_missing_rates)

    structural_mismatch = frame.copy()
    structural_mismatch.loc[0, "conf__native__sigma_tvt"] = np.nan
    with pytest.raises(ValueError, match="structural NaN rate"):
        validate_inference_feature_missingness(structural_mismatch, training_missing_rates)

    infinite = frame.copy()
    infinite.loc[0, "formula__component_std"] = np.inf
    with pytest.raises(ValueError, match=r"\+/-inf"):
        validate_inference_feature_missingness(infinite, training_missing_rates)


def test_exp264_stage_a_catalog_drives_real_selector_missingness_contract() -> None:
    config = yaml.safe_load((EXP / "config.yaml").read_text())
    catalog_path = EXP / "kaggle/output/stage_a_v4/artifacts/feature_catalog.csv"
    require_saved_files(catalog_path)
    assert sha256_file(catalog_path) == config["inference"][
        "selector_feature_catalog_sha256"
    ]
    catalog = pd.read_csv(catalog_path)
    selected = catalog.loc[catalog["selected"].astype(str).str.lower().eq("true")].copy()
    selected["missing_rate"] = pd.to_numeric(selected["missing_rate"], errors="raise")
    assert len(selected) == 88
    assert selected["missing_rate"].gt(0).sum() == 25

    frame = pd.DataFrame(
        np.ones((12, len(selected)), dtype=np.float32),
        columns=selected["feature"].astype(str),
    )
    for row in selected.itertuples(index=False):
        if not str(row.feature).startswith(("conf__", "formula__")):
            continue
        missing_rows = int(round(float(row.missing_rate) * len(frame)))
        assert np.isclose(missing_rows / len(frame), float(row.missing_rate))
        if missing_rows:
            frame.loc[: missing_rows - 1, str(row.feature)] = np.nan
    missing_rates = dict(
        zip(
            selected["feature"].astype(str),
            selected["missing_rate"].astype(float),
            strict=True,
        )
    )
    summary = validate_inference_feature_missingness(frame, missing_rates)
    assert summary["structural_missingness"].sum() == selected[
        "feature"
    ].astype(str).str.startswith(("conf__", "formula__")).sum()


def test_nested_inner_fold_maps_are_well_disjoint_and_deterministic() -> None:
    fold_well_counts = {
        outer_fold: pd.DataFrame(
            {
                "well": [f"fold{outer_fold}_well{index}" for index in range(5)],
                "rows": [100 + 10 * outer_fold + index for index in range(5)],
            }
        )
        for outer_fold in range(5)
    }
    first, first_manifest = build_nested_inner_fold_maps(
        fold_well_counts, n_outer_folds=5, n_inner_folds=4
    )
    second, second_manifest = build_nested_inner_fold_maps(
        fold_well_counts, n_outer_folds=5, n_inner_folds=4
    )
    assert first == second
    pd.testing.assert_frame_equal(first_manifest, second_manifest)
    assert len(first_manifest) == 20
    for downstream_outer_fold in range(5):
        outer_valid_wells = set(fold_well_counts[downstream_outer_fold]["well"])
        expected_train_wells = set().union(
            *(
                set(frame["well"])
                for source_fold, frame in fold_well_counts.items()
                if source_fold != downstream_outer_fold
            )
        )
        assignment = first[downstream_outer_fold]
        assert set(assignment) == expected_train_wells
        assert not outer_valid_wells.intersection(assignment)
        assert set(assignment.values()) == {0, 1, 2, 3}


def test_stage_d_cost_contract_is_exactly_30_gpu_boosters() -> None:
    config = yaml.safe_load((EXP / "config.yaml").read_text())
    contract = stage_d_cost_contract(config)
    stage = config["model"]["downstream_tvt_stage"]
    assert stage["feature_surface"] == "exp218_clean_273_drop_107"
    assert stage["expected_source_base_feature_count"] == 380
    assert stage["expected_base_feature_count"] == 273
    assert stage["matched_control_feature_count"] == 273
    assert stage["selector_compact_addonly_feature_count"] == 347
    assert contract["variants"] == ["matched_control", "selector_compact_addonly"]
    assert contract["lightgbm_config_indices"] == [0, 1, 2]
    assert contract["folds"] == 5
    assert contract["boosters_per_variant"] == 15
    assert contract["total_gpu_boosters"] == 30
    assert contract["approval_received_at"] == "2026-07-18"
    assert contract["approval_scope"] == (
        "clean273_control15_compact347_addonly15_three_configs_five_folds_30_gpu_boosters"
    )
    broken = yaml.safe_load((EXP / "config.yaml").read_text())
    broken["model"]["downstream_tvt_stage"]["planned_gpu_boosters"] = 15
    with pytest.raises(ValueError, match="30 boosters"):
        stage_d_cost_contract(broken)
    stale_approval = yaml.safe_load((EXP / "config.yaml").read_text())
    stale_approval["model"]["downstream_tvt_stage"][
        "corrected_run_approval_received_at"
    ] = None
    with pytest.raises(ValueError, match="corrected Stage D approval is missing"):
        stage_d_cost_contract(stale_approval)


def test_stage_d_clean_273_allowlist_is_sha_locked_and_fold_safe() -> None:
    config = yaml.safe_load((EXP / "config.yaml").read_text())
    stage = config["model"]["downstream_tvt_stage"]
    availability_path = (
        EXP / "artifacts/feature_availability_audit/exp218_feature_availability.csv"
    )
    allowlist_path = EXP / "artifacts/feature_availability_audit/exp218_clean_273_allowlist.csv"
    require_saved_files(availability_path, allowlist_path)
    availability = pd.read_csv(availability_path)
    selected, evidence = apply_stage_d_base_feature_allowlist(
        availability["feature"].astype(str).tolist(),
        allowlist_path=allowlist_path,
        expected_source_count=stage["expected_source_base_feature_count"],
        expected_selected_count=stage["expected_base_feature_count"],
        expected_allowlist_sha256=stage["base_feature_allowlist_sha256"],
    )
    safe = availability.set_index("feature").loc[selected]
    assert len(selected) == 273
    assert safe["fold_safe"].all()
    assert safe["hidden_safe"].all()
    assert evidence["dropped_feature_count"] == 107


def test_stage_d_clean_273_allowlist_rejects_sha_mismatch() -> None:
    availability_path = (
        EXP / "artifacts/feature_availability_audit/exp218_feature_availability.csv"
    )
    allowlist_path = EXP / "artifacts/feature_availability_audit/exp218_clean_273_allowlist.csv"
    require_saved_files(availability_path, allowlist_path)
    availability = pd.read_csv(availability_path)
    with pytest.raises(ValueError, match="allowlist SHA mismatch"):
        apply_stage_d_base_feature_allowlist(
            availability["feature"].astype(str).tolist(),
            allowlist_path=allowlist_path,
            expected_source_count=380,
            expected_selected_count=273,
            expected_allowlist_sha256="0" * 64,
        )


def test_stage_d_guard_requires_matched_non_regression_checks() -> None:
    guard_config = {
        "min_improved_folds": 3,
        "max_near_delta_rmse": 0.0,
        "max_1000_plus_delta_rmse": 0.0,
        "max_hidden_like_delta_rmse": 0.0,
        "max_worst_well_regression": 0.25,
    }
    passed = stage_d_matched_guard(
        pooled_delta_rmse=-0.03,
        fold_deltas=[-0.1, -0.02, 0.01, -0.03, 0.02],
        near_delta_rmse=-0.01,
        distance_1000_plus_delta_rmse=-0.02,
        hidden_like_deltas=[-0.01, -0.03],
        worst_well_delta_rmse=0.2,
        guard_config=guard_config,
    )
    assert passed["passed"] is True
    assert passed["improved_folds"] == 3
    failed = stage_d_matched_guard(
        pooled_delta_rmse=-0.03,
        fold_deltas=[-0.1, -0.02, 0.01, -0.03, 0.02],
        near_delta_rmse=0.001,
        distance_1000_plus_delta_rmse=-0.02,
        hidden_like_deltas=[-0.01, -0.03],
        worst_well_delta_rmse=0.2,
        guard_config=guard_config,
    )
    assert failed["passed"] is False
    assert failed["checks"]["near_non_regression"] is False


def test_corrected_saved_model_inference_and_external_submit_override_is_explicit() -> None:
    config = yaml.safe_load((EXP / "config.yaml").read_text())
    inference = config["inference"]
    assert config["execution"]["stage"] == "hidden_safe_inference"
    expected_scope = (
        "clean273_control15_compact347_addonly15_three_configs_five_folds_30_gpu_boosters"
    )
    assert config["execution"]["approval_scope"] == expected_scope
    assert config["model"]["downstream_tvt_stage"][
        "corrected_run_approval_scope"
    ] == expected_scope
    assert config["execution"]["run_approved"] is False
    assert config["execution"]["inference_enabled"] is True
    assert inference["status"] == "corrected_inference_v4_complete_public_lb_7p562"
    assert inference["stage_d_guard_passed"] is False
    assert inference["retained_guard_failure"] == (
        "worst_well_regression_plus_14p482873"
    )
    assert inference["expected_selector_feature_count"] == 88
    assert inference["selector_training_sparse_feature_count"] == 25
    assert inference["expected_source_base_feature_count"] == 380
    assert inference["expected_base_feature_count"] == 273
    assert inference["expected_compact_feature_count"] == 74
    assert inference["expected_final_feature_count"] == 347
    assert inference["submit_to_kaggle"] is False
    assert inference["competition_submit_authorized"] is True
    assert inference["competition_submit_requires_submit_check_pass"] is True
    assert inference["competition_submit_status"] == "complete_public_lb_7p562"
    assert inference["competition_submission_ref"] == 54818932
    assert inference["competition_submission_public_score"] == pytest.approx(7.562)
    assert inference["competition_submission_status_final"] == "SubmissionStatus.COMPLETE"
    assert inference["kaggle_status_final"] == "KernelWorkerStatus.COMPLETE"
    assert config["features"]["raw_context"]["horizontal_numeric_allowlist"] == [
        "MD",
        "X",
        "Y",
        "Z",
        "GR",
    ]


def test_stage_c_root_verification_requires_complete_frozen_partitions(
    tmp_path: Path,
) -> None:
    root = tmp_path / "stage_c" / "artifacts"
    root.mkdir(parents=True)
    schema = {
        "schema_version": "1.0.0",
        "features": [f"selector__f{index}" for index in range(74)],
        "compact_meta_schema_sha256": "logical-schema-sha",
    }
    metrics = {
        "model_count": 40,
        "score_guard": {"passed": True},
        "leakage_audit": {"passed": True},
    }
    model_manifest = {"model_count": 40}
    partitions = []
    total_rows = 0
    for downstream_fold in range(5):
        for source_fold in range(5):
            role = "valid" if source_fold == downstream_fold else "train"
            relative = Path("nested_compact_meta") / str(downstream_fold) / role / str(
                source_fold
            ) / "part.parquet"
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            rows = 756_797 + (1 if len(partitions) < 20 else 0)
            path.write_text(f"partition-{downstream_fold}-{source_fold}")
            total_rows += rows
            partitions.append(
                {
                    "downstream_outer_fold": downstream_fold,
                    "role": role,
                    "source_outer_fold": source_fold,
                    "rows": rows,
                    "path": str(relative),
                    "sha256": sha256_file(path),
                }
            )
    assert total_rows == 18_919_945
    compact_manifest = {
        "partition_count": 25,
        "rows": total_rows,
        "compact_meta_schema_sha256": "logical-schema-sha",
        "partitions": partitions,
    }
    contract_files = {
        "compact_meta_schema.json": schema,
        "nested_selector_metrics.json": metrics,
        "nested_selector_model_manifest.json": model_manifest,
        "nested_compact_manifest.json": compact_manifest,
    }
    for filename, payload in contract_files.items():
        (root / filename).write_text(json.dumps(payload))
    config = {
        "data": {
            "stage_c_artifact_root_patterns": [str(root)],
            "stage_c_expected_compact_meta_schema_file_sha256": sha256_file(
                root / "compact_meta_schema.json"
            ),
            "stage_c_expected_compact_meta_schema_logical_sha256": "logical-schema-sha",
        }
    }
    assert resolve_stage_c_artifact_root(config, [tmp_path]) == root
    evidence = verify_stage_c_artifact_root(root, config)
    assert evidence["partition_count"] == 25
    assert evidence["compact_feature_count"] == 74
    (root / partitions[0]["path"]).write_text("tampered")
    with pytest.raises(ValueError, match="partition SHA mismatch"):
        verify_stage_c_artifact_root(root, config)

    logical_mismatch = {
        "data": {
            **config["data"],
            "stage_c_expected_compact_meta_schema_logical_sha256": "wrong-logical-sha",
        }
    }
    (root / partitions[0]["path"]).write_text("partition-0-0")
    with pytest.raises(ValueError, match="logical SHA mismatch"):
        verify_stage_c_artifact_root(root, logical_mismatch)
