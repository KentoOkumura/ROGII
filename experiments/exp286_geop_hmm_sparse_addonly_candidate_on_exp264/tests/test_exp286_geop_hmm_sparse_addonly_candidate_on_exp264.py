from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest
import yaml

from src.candidate_selector_pipeline import FoldBundle, contract_by_id
from src.geop_hmm_selector_audit import (
    _normalize_stage_c_fold_metrics,
    augment_fold_bundle,
    build_base_contract,
    load_parent_stage_d_reference,
    raw_test_only_schema_guard,
    stage_d_full13_cost_contract,
    validate_full_contract,
)


ROOT = Path(__file__).resolve().parents[3]
EXP = "exp286_geop_hmm_sparse_addonly_candidate_on_exp264"
EXP_DIR = ROOT / "experiments" / EXP
TRAIN_SOURCE = EXP_DIR / f"{EXP}_train.py"


def full_candidate_contract() -> dict:
    return yaml.safe_load((EXP_DIR / "candidate_contract.yaml").read_text())


def synthetic_base_bundle() -> FoldBundle:
    contract = build_base_contract(full_candidate_contract())
    ids = [str(item["id"]) for item in contract["score_candidates"]]
    base = pd.DataFrame(
        {
            "id": ["well_a_10", "well_b_20"],
            "well": ["well_a", "well_b"],
            "well_row_idx": [10, 20],
            "outer_fold": np.array([0, 1], dtype=np.int8),
            "md_since": np.array([100.0, 1200.0], dtype=np.float32),
            "last_known_tvt": np.array([1000.0, 2000.0], dtype=np.float32),
        }
    )
    values = np.arange(24, dtype=np.float32).reshape(2, 12) + 1000.0
    return FoldBundle(
        base=base,
        values=values,
        available=np.ones_like(values, dtype=bool),
        confidence={},
        candidate_ids=ids,
        specs=contract_by_id(contract),
    )


def synthetic_geop_source() -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "id": ["well_a_10", "well_b_20"],
            "well": ["well_a", "well_b"],
            "row_idx": [10, 20],
            "fold": np.array([0, 1], dtype=np.int8),
            "geop_hmm": np.array([1001.0, 2001.0], dtype=np.float32),
            "geop_hmm_std": np.array([1.5, 2.5], dtype=np.float32),
            "geop_hmm_loglik": np.array([-12.0, -20.0], dtype=np.float32),
            "evaluation_rows_in_well": np.array([2, 4], dtype=np.int32),
            "loglik_per_row": np.array([-6.0, -5.0], dtype=np.float32),
        }
    )
    return frame.set_index("id", drop=False)


def load_train_module() -> ModuleType:
    os.environ["EXP286_IMPORT_ONLY"] = "1"
    spec = importlib.util.spec_from_file_location("exp286_train_for_test", TRAIN_SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def module() -> ModuleType:
    return load_train_module()


def gate_config() -> dict:
    return {
        "gate": {
            "cutoff_fraction": 0.25,
            "missing_rank_value": 0.5,
            "forbidden_tokens": [
                "true_tvt",
                "target",
                "error",
                "oracle",
                "geop_hmm",
            ],
            "features": [
                {"name": "geometry_quality", "direction": "higher"},
                {"name": "bank_stress", "direction": "higher"},
            ],
        }
    }


def runtime_config(iterations: int = 64) -> dict:
    return {
        "runtime_guard": {
            "shadow_wells": 200,
            "selected_fraction_max": 0.25,
            "selected_wells_max": 50,
            "per_fold_fraction_max": 0.30,
            "geop_additional_p95_seconds_max": 2700.0,
            "total_p95_seconds_max": 27000.0,
            "bootstrap_iterations": iterations,
            "seed": 42,
            "required_columns": [
                "well",
                "fold",
                "gate_selected",
                "base_seconds",
                "geop_additional_seconds",
                "selector_seconds",
                "tvt_model_seconds",
                "save_seconds",
            ],
            "forbidden_columns": [
                "TVT",
                "TVT_input",
                "true_tvt",
                "target",
                "error",
                "abs_error",
                "oracle",
            ],
        }
    }


def test_forbidden_gate_features_are_rejected(module: ModuleType) -> None:
    with pytest.raises(ValueError, match="forbidden gate features"):
        module.reject_forbidden_gate_features(
            ["geometry_quality", "candidate_error_signal"],
            ["error", "oracle", "geop_hmm"],
        )


def test_fixed_rank_gate_is_deterministic_and_fold_bounded(module: ModuleType) -> None:
    rows = []
    for fold in range(5):
        for position in range(8):
            rows.append(
                {
                    "well": f"f{fold}_w{position}",
                    "outer_fold": fold,
                    "geometry_quality": float(position),
                    "bank_stress": float(position),
                }
            )
    frame = pd.DataFrame(rows)
    first = module.apply_fixed_rank_gate(frame, gate_config())
    second = module.apply_fixed_rank_gate(frame.sample(frac=1.0, random_state=7), gate_config())
    first_selected = set(first.loc[first["candidate_available"], "well"])
    second_selected = set(second.loc[second["candidate_available"], "well"])
    assert first_selected == second_selected
    assert len(first_selected) == 10
    assert all(first.groupby("outer_fold")["candidate_available"].sum().eq(2))
    assert all(name.endswith(("w6", "w7")) for name in first_selected)


def test_fixed_rank_gate_excludes_boundary_ties_without_well_id_rule(
    module: ModuleType,
) -> None:
    frame = pd.DataFrame(
        {
            "well": [f"w{position}" for position in range(8)],
            "outer_fold": 0,
            "geometry_quality": [0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 6.0, 6.0],
            "bank_stress": [0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 6.0, 6.0],
        }
    )
    result = module.apply_fixed_rank_gate(frame, gate_config())
    assert int(result["candidate_available"].sum()) == 0


def test_add_only_oracle_never_worsens_and_sparse_respects_gate(
    module: ModuleType,
) -> None:
    arrays = module.OracleArrays(
        existing_squared_error=np.asarray(
            [
                [4.0, 9.0],
                [4.0, 1.0],
                [9.0, 16.0],
                [9.0, 4.0],
            ],
            dtype=np.float64,
        ),
        geop_squared_error=np.asarray([1.0, 9.0, 1.0, 1.0], dtype=np.float64),
        gate_row_available=np.asarray([True, True, False, False]),
        well_codes=np.asarray([0, 0, 1, 1], dtype=np.int32),
        block_codes=np.asarray([0, 0, 1, 1], dtype=np.int32),
        well_names=np.asarray(["a", "b"], dtype=object),
        well_fold=np.asarray([0, 1], dtype=np.int8),
        well_gate=np.asarray([True, False]),
    )
    mask = np.ones(4, dtype=bool)
    row = module.oracle_metrics_for_mask(
        arrays, mask, scope="overall", granularity="row"
    )
    whole = module.oracle_metrics_for_mask(
        arrays, mask, scope="overall", granularity="whole_well"
    )
    assert row["full_union_oracle_rmse"] <= row["base_oracle_rmse"]
    assert row["sparse_union_oracle_rmse"] <= row["base_oracle_rmse"]
    assert whole["full_union_oracle_rmse"] <= whole["base_oracle_rmse"]
    assert whole["sparse_union_oracle_rmse"] <= whole["base_oracle_rmse"]
    assert whole["full_sse_gain"] > whole["sparse_sse_gain"]


def test_unique_best_reports_full_and_gated_counts(module: ModuleType) -> None:
    arrays = module.OracleArrays(
        existing_squared_error=np.asarray([[4.0], [4.0], [9.0]], dtype=np.float64),
        geop_squared_error=np.asarray([1.0, 5.0, 1.0], dtype=np.float64),
        gate_row_available=np.asarray([True, True, False]),
        well_codes=np.asarray([0, 0, 1], dtype=np.int32),
        block_codes=np.asarray([0, 0, 1], dtype=np.int32),
        well_names=np.asarray(["a", "b"], dtype=object),
        well_fold=np.asarray([0, 1], dtype=np.int8),
        well_gate=np.asarray([True, False]),
    )
    result = module.unique_best_for_mask(
        arrays,
        np.ones(3, dtype=bool),
        scope="overall",
        granularity="row",
        tolerance=1e-9,
    )
    assert result["unique_best_groups"] == 2
    assert result["gated_unique_best_groups"] == 1


def test_runtime_guard_is_fail_closed_without_shadow_manifest(module: ModuleType) -> None:
    result = module.evaluate_runtime_guard(None, runtime_config())
    assert result["passed"] is False
    assert result["status"].startswith("not_evaluated")


def test_runtime_guard_passes_bounded_200_well_manifest(module: ModuleType) -> None:
    wells = np.arange(200)
    frame = pd.DataFrame(
        {
            "well": [f"w{value:03d}" for value in wells],
            "fold": wells % 5,
            "gate_selected": (wells % 4) == 0,
            "base_seconds": 30.0,
            "geop_additional_seconds": 5.0,
            "selector_seconds": 2.0,
            "tvt_model_seconds": 3.0,
            "save_seconds": 1.0,
        }
    )
    result = module.evaluate_runtime_guard(frame, runtime_config())
    assert result["passed"] is True
    assert result["selected_wells"] == 50
    assert result["geop_additional_p95_seconds"] < 2700.0
    assert result["total_p95_seconds"] < 27000.0


def test_runtime_guard_rejects_truth_columns(module: ModuleType) -> None:
    frame = pd.DataFrame(
        {
            "well": [f"w{value:03d}" for value in range(200)],
            "fold": np.arange(200) % 5,
            "gate_selected": False,
            "base_seconds": 1.0,
            "geop_additional_seconds": 0.0,
            "selector_seconds": 0.0,
            "tvt_model_seconds": 0.0,
            "save_seconds": 0.0,
            "true_tvt": 12000.0,
        }
    )
    with pytest.raises(ValueError, match="forbidden columns"):
        module.evaluate_runtime_guard(frame, runtime_config())


def test_full13_candidate_contract_preserves_ids_domains_and_confidence() -> None:
    contract = full_candidate_contract()
    evidence = validate_full_contract(contract)
    assert evidence["candidate_count"] == 13
    assert evidence["candidate_order"][-1] == "geop_hmm"
    assert evidence["geop_hmm_spec"]["kind"] == "primitive"
    assert evidence["geop_hmm_spec"]["family"] == "geop_centered_exact_hmm"
    assert contract["candidate_id_model_encoding"] == {
        "type": "one_hot",
        "order_source": "score_candidates_declared_order",
        "width": 13,
        "keep_string_id_in_artifacts": True,
        "ordinal_index_as_model_feature": False,
    }
    assert len(contract["legal_domains"]["primitive_pair_bank"]["candidates"]) == 12
    assert len(contract["legal_domains"]["primitive_fixed_bank"]["candidates"]) == 8
    for domain in contract["legal_domains"].values():
        assert "geop_hmm" in domain["candidates"]
    required = contract["confidence_contract"]["current_test_required_fields_by_primitive"][
        "geop_hmm"
    ]
    assert required == [
        "confidence_valid",
        "sigma_tvt",
        "source_loglik",
        "loglik_per_row",
        "candidate_finite_source",
    ]


def test_geop_bundle_appends_full_available_candidate_and_native_confidence() -> None:
    contract = full_candidate_contract()
    bundle = augment_fold_bundle(synthetic_base_bundle(), synthetic_geop_source(), contract)
    assert bundle.values.shape == (2, 13)
    assert bundle.candidate_ids[-1] == "geop_hmm"
    assert np.array_equal(bundle.values[:, -1], np.array([1001.0, 2001.0], np.float32))
    assert bundle.available[:, -1].all()
    confidence = bundle.confidence["geop_hmm"]
    assert confidence["confidence_valid"].all()
    assert np.array_equal(confidence["sigma_tvt"], np.array([1.5, 2.5], np.float32))
    assert np.array_equal(confidence["source_loglik"], np.array([-12.0, -20.0], np.float32))
    assert np.array_equal(confidence["loglik_per_row"], np.array([-6.0, -5.0], np.float32))
    assert confidence["candidate_finite_source"].eq(1).all()
    assert not any("target" in column.lower() for column in confidence.columns)
    assert not any("error" in column.lower() for column in confidence.columns)
    assert not any("oracle" in column.lower() for column in confidence.columns)
    assert not any("true_tvt" in column.lower() for column in confidence.columns)


def test_geop_bundle_join_fails_closed_on_fold_mismatch() -> None:
    source = synthetic_geop_source().copy()
    source.loc["well_b_20", "fold"] = 2
    with pytest.raises(ValueError, match="outer-fold alignment"):
        augment_fold_bundle(synthetic_base_bundle(), source, full_candidate_contract())


def test_geop_bundle_uses_exp263_outer_fold_not_exp279_source_fold() -> None:
    source = synthetic_geop_source().copy()
    source["outer_fold"] = np.array([0, 1], dtype=np.int8)
    source["fold"] = np.array([4, 3], dtype=np.int8)
    bundle = augment_fold_bundle(
        synthetic_base_bundle(), source, full_candidate_contract()
    )
    assert bundle.available[:, -1].all()
    assert np.array_equal(bundle.values[:, -1], np.array([1001.0, 2001.0], np.float32))


def test_raw_test_only_schema_guard_rejects_training_only_formation() -> None:
    config = yaml.safe_load((EXP_DIR / "config.yaml").read_text())
    passed = raw_test_only_schema_guard(
        {"features": ["ctx__raw__md", "id__candidate__geop_hmm"]}, config
    )
    assert passed["passed"] is True
    with pytest.raises(ValueError, match="training-only context"):
        raw_test_only_schema_guard({"features": ["ctx__raw__ancc"]}, config)


def test_repository_config_enables_exactly_approved_stage_d_scope() -> None:
    config = yaml.safe_load((EXP_DIR / "config.yaml").read_text())
    assert config["experiment"]["route"] == "ensemble"
    assert config["candidate_bank"]["existing_candidate_count"] == 12
    assert config["candidate_bank"]["selector_candidate_count"] == 13
    assert config["candidate_bank"]["added_candidate"]["id"] == "geop_hmm"
    assert config["candidate_bank"]["added_candidate"]["kind"] == "primitive"
    assert config["candidate_bank"]["fixed_gate_enabled"] is False
    execution = config["execution"]
    assert execution["stage"] == "downstream_tvt_addonly_full13"
    assert execution["active_variants"] == 1
    assert execution["lightgbm_config_count"] == 3
    assert execution["objectives"] == 1
    assert execution["trained_fold_count"] == 5
    assert execution["total_boosters"] == 15
    assert execution["hmm_well_runs"] == 0
    assert execution["pf_well_runs"] == 0
    assert execution["parent_control_retraining"] is False
    assert execution["stage_b_enabled"] is False
    assert execution["stage_c_enabled"] is False
    assert execution["stage_d_enabled"] is True
    assert execution["gpu"] is True
    assert config["inference"]["enabled"] is False
    module = load_train_module()
    package_config = yaml.safe_load((EXP_DIR / "config.yaml").read_text())
    package_config["execution"]["run_approved"] = True
    contract = module.stage_d_execution_contract(package_config)
    assert contract["outer_folds"] == 5
    assert contract["total_gpu_boosters"] == 15
    assert contract["parent_control_retraining"] is False


def test_stage_d_scope_is_addonly_only_15_gpu_models_without_control_retraining() -> None:
    config = yaml.safe_load((EXP_DIR / "config.yaml").read_text())
    cost = stage_d_full13_cost_contract(config)
    assert cost["variants"] == ["selector_compact_addonly"]
    assert cost["lightgbm_config_indices"] == [0, 1, 2]
    assert cost["folds"] == 5
    assert cost["total_gpu_boosters"] == 15
    assert cost["control_retraining"] is False
    stage = config["model"]["downstream_tvt_stage"]
    assert stage["expected_base_feature_count"] == 273
    assert stage["expected_compact_feature_count"] == 77
    assert stage["selector_compact_addonly_feature_count"] == 350
    data = config["data"]
    assert data["stage_c_expected_nested_selector_metrics_sha256"] == (
        "8f69a1a45ad467007c4fba22b0470e216c1864ac41834e579f8fe88b88381d61"
    )
    assert data["stage_c_expected_compact_meta_schema_logical_sha256"] == (
        "73e7efd53b8823ace6e35e8caf0e2f7cd214213e142888f6576ba89a774310ec"
    )


def test_parent_stage_d_reference_is_sha_locked() -> None:
    config = yaml.safe_load((EXP_DIR / "config.yaml").read_text())
    root = (
        ROOT
        / "experiments/exp264_exp263_candidate_confidence_dual_selector"
        / "kaggle/output/stage_d_v3_corrected/artifacts"
    )
    reference = load_parent_stage_d_reference(
        config=config,
        paths={
            "metrics": root / "stage_d_metrics.json",
            "fold_metrics": root / "stage_d_fold_metrics.csv",
            "bucket_metrics": root / "stage_d_bucket_metrics.csv",
            "hidden_like_metrics": root / "stage_d_hidden_like_metrics.csv",
            "by_well": root / "stage_d_by_well.csv",
        },
    )
    assert reference["metrics"]["selector_compact_addonly_lgb_mean_rmse"] == pytest.approx(
        8.460811237612477
    )
    assert len(reference["by_well"]) == 773


def test_stage_c_fold_metrics_accept_parent_fold_column() -> None:
    frame = pd.DataFrame(
        {
            "fold": [4, 2, 0, 3, 1],
            "hard_primary_rmse": [8.4, 8.2, 8.0, 8.3, 8.1],
        }
    )
    normalized = _normalize_stage_c_fold_metrics(frame, label="parent")
    assert normalized["outer_fold"].tolist() == [0, 1, 2, 3, 4]
    assert normalized["hard_primary_rmse"].tolist() == [8.0, 8.1, 8.2, 8.3, 8.4]


def test_notebook_sources_are_not_file_dependent_and_train_uses_shared_heavy_pipeline() -> None:
    inference = EXP_DIR / f"{EXP}_inference.py"
    for path in [TRAIN_SOURCE, inference]:
        source = path.read_text()
        assert "Path(__file__)" not in source
        assert "from settings import" not in source
    train_source = TRAIN_SOURCE.read_text()
    assert "from src.geop_hmm_selector_audit import" in train_source
    assert "run_geop_hmm_selector_stage_b" in train_source
    assert "run_geop_hmm_selector_stage_c" in train_source
    assert "run_geop_hmm_stage_d_addonly" in train_source
    helper_source = (ROOT / "src/geop_hmm_selector_audit.py").read_text()
    assert 'stage_c_evidence["sha256"][' in helper_source
    assert 'stage_c_evidence["nested_selector_model_manifest_sha256"]' not in helper_source
    inference_source = inference.read_text()
    assert "from src" not in inference_source
