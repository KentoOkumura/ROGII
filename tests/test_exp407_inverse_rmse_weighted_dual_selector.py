from __future__ import annotations

import json
import runpy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.candidate_selector_pipeline import read_yaml
from src.candidate_task_weighting import (
    build_inverse_rmse_candidate_task_weights,
    write_candidate_task_weight_artifacts,
)
from src.exp407_inverse_rmse_selector import (
    evaluate_scientific_gate,
    resolve_pinned_input,
    validate_exp407_static_contract,
)


ROOT = Path(__file__).resolve().parents[1]
EXP_NAME = "exp407_fold_safe_inverse_rmse_weighted_dual_selector_on_exp264"
EXP_DIR = ROOT / "experiments" / EXP_NAME
CONFIG_PATH = EXP_DIR / "config.yaml"
CONTRACT_PATH = EXP_DIR / "candidate_contract.yaml"
NOTEBOOK_SOURCE = EXP_DIR / f"{EXP_NAME}_compact_selfcontained_train.py"


def load_config() -> dict:
    with CONFIG_PATH.open() as handle:
        value = yaml.safe_load(handle)
    assert isinstance(value, dict)
    return value


def weight_config() -> dict:
    return load_config()["candidate_task_weight"]


def make_labels(
    candidate_order: list[str],
    error_rows: list[list[float]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for base_position, errors in enumerate(error_rows):
        for candidate_id, error in zip(candidate_order, errors, strict=True):
            rows.append(
                {
                    "id": f"row_{base_position}",
                    "well": f"well_{base_position // 2}",
                    "well_row_idx": base_position,
                    "outer_fold": base_position % 2,
                    "candidate_id": candidate_id,
                    "candidate_abs_error": error,
                }
            )
    return pd.DataFrame(rows)


def test_static_contract_freezes_cost_candidates_and_closed_execution_boundary() -> None:
    config = load_config()
    contract = read_yaml(CONTRACT_PATH)
    audit = validate_exp407_static_contract(config, contract)

    assert config["experiment"]["route"] == "ml_model"
    assert config["implementation"]["stage_b_implemented"] is True
    assert config["execution"]["run_approved"] is False
    assert config["execution"]["run_completed"] is True
    assert config["stages"]["stage_b"]["enabled"] is False
    assert config["outcome"]["decision"] == "fail_close_exp407_without_rescue"
    assert config["execution"]["approval_scope"] == (
        "inverse_rmse_weighted_stage_b_1_variant_2_objectives_"
        "5_outer_10_cpu_boosters_no_control_retraining"
    )
    assert audit["candidate_count"] == 12
    assert audit["legal_domain_counts"] == [11, 7]
    assert audit["feature_count"] == 88
    assert audit["cost"] == {
        "active_variants": 1,
        "objectives": 2,
        "outer_folds": 5,
        "planned_cpu_boosters": 10,
        "parent_control_retraining": False,
        "gpu_boosters": 0,
        "pf_hmm_beam_regeneration": False,
        "inference": False,
        "submission": False,
    }


def test_inverse_rmse_weights_use_only_fit_labels_and_preserve_mean_one() -> None:
    candidates = ["a", "b", "c"]
    labels = make_labels(
        candidates,
        [
            [2.0, 3.0, 4.0],
            [2.0, 3.0, 4.0],
            [2.0, 3.0, 4.0],
            [2.0, 3.0, 4.0],
        ],
    )
    result = build_inverse_rmse_candidate_task_weights(
        labels,
        candidates,
        partition_id=4,
        config=weight_config(),
    )

    raw = 1.0 / np.asarray([2.0, 3.0, 4.0])
    expected = raw / raw.mean()
    assert np.allclose(result.table["fit_candidate_rmse"], [2.0, 3.0, 4.0])
    assert np.allclose(result.table["final_weight"], expected)
    assert np.allclose(result.sample_weight.reshape(-1, 3), expected)
    assert result.audit["final_weight_mean"] == pytest.approx(1.0, abs=1.0e-12)
    assert result.truth_read_ledger["outer_valid_truth_reads_for_weight"] == 0
    assert result.truth_read_ledger["global_oof_truth_reads_for_weight"] == 0
    assert result.sampling_manifest["fit_base_rows"] == 4
    assert result.sampling_manifest["fit_long_rows"] == 12


def test_post_clip_renormalization_range_violation_fails_closed() -> None:
    candidates = ["a", "b", "c"]
    labels = make_labels(candidates, [[1.0, 2.0, 4.0]] * 3)
    with pytest.raises(ValueError, match="outside"):
        build_inverse_rmse_candidate_task_weights(
            labels,
            candidates,
            partition_id=0,
            config=weight_config(),
        )


def test_candidate_order_and_complete_blocks_are_required() -> None:
    candidates = ["a", "b", "c"]
    labels = make_labels(candidates, [[2.0, 3.0, 4.0]] * 2)
    wrong_order = labels.copy()
    wrong_order.loc[0, "candidate_id"] = "b"
    with pytest.raises(ValueError, match="candidate-long order"):
        build_inverse_rmse_candidate_task_weights(
            wrong_order,
            candidates,
            partition_id=0,
            config=weight_config(),
        )
    with pytest.raises(ValueError, match="complete base-row blocks"):
        build_inverse_rmse_candidate_task_weights(
            labels.iloc[:-1],
            candidates,
            partition_id=0,
            config=weight_config(),
        )


def test_weight_artifacts_record_sampling_truth_and_shared_objective_contract(
    tmp_path: Path,
) -> None:
    candidates = ["a", "b", "c"]
    results = []
    for partition in (0, 1):
        labels = make_labels(candidates, [[2.0, 3.0, 4.0]] * 4)
        result = build_inverse_rmse_candidate_task_weights(
            labels,
            candidates,
            partition_id=partition,
            config=weight_config(),
        )
        result.audit["fit_valid_well_overlap"] = 0
        results.append(result)

    manifest = write_candidate_task_weight_artifacts(tmp_path, results)
    assert manifest["all_checks_passed"] is True
    assert manifest["same_weight_for_objectives"] == [
        "pred_abs_error",
        "p_within10",
    ]
    assert manifest["truth_read_ledger"]["forbidden_truth_reads"] == 0
    assert (tmp_path / "candidate_task_weight_by_fold.csv").is_file()
    assert (tmp_path / "candidate_task_weight_sampling_manifest.csv").is_file()
    assert (tmp_path / "candidate_task_weight_truth_read_ledger.csv").is_file()
    saved = json.loads((tmp_path / "candidate_task_weight_manifest.json").read_text())
    assert saved["validation_sample_weight_applied"] is False
    assert saved["metric_sample_weight_applied"] is False


def test_pinned_input_rejects_wrong_sha(tmp_path: Path) -> None:
    path = tmp_path / "input.json"
    path.write_text("{}\n")
    with pytest.raises(FileNotFoundError, match="pinned SHA"):
        resolve_pinned_input(
            [str(path)],
            [tmp_path],
            expected_sha256="0" * 64,
            label="synthetic input",
        )


def test_scientific_gate_is_all_and_and_closes_on_one_tail_failure() -> None:
    config = load_config()
    parent_fold = pd.DataFrame(
        {
            "expected_error_mae": [3.80] * 5,
            "within10_logloss": [0.36] * 5,
            "within10_brier": [0.112] * 5,
            "hard_primary_rmse": [8.60] * 5,
        }
    )
    new_fold = pd.DataFrame(
        {
            "expected_error_mae": [3.78] * 5,
            "within10_logloss": [0.359] * 5,
            "within10_brier": [0.111] * 5,
            "hard_primary_rmse": [8.58] * 5,
        }
    )
    new_metrics = {
        "pooled_score_metrics": {
            "expected_error_mae": 3.78,
            "within10_logloss": 0.359,
            "within10_brier": 0.111,
        },
        "hard_primary_oof_rmse": 8.58,
    }
    pooled_bucket = {
        "near_0_250__delta_new_minus_parent": 0.0,
        "1000_plus__delta_new_minus_parent": 0.0,
    }
    hidden = pd.DataFrame(
        {
            "scope": [
                "hidden_like_spatial",
                "hidden_like_typewell_purged",
            ],
            "delta_rmse_new_minus_parent": [0.0, 0.0],
        }
    )
    by_well = pd.DataFrame({"delta_rmse_new_minus_parent": [0.10, -0.20]})
    passed = evaluate_scientific_gate(
        new_metrics=new_metrics,
        new_fold=new_fold,
        parent_fold=parent_fold,
        pooled_bucket=pooled_bucket,
        hidden=hidden,
        by_well=by_well,
        gate=config["gates"]["scientific"],
    )
    assert passed["passed"] is True
    assert all(passed["checks"].values())

    failed_by_well = pd.DataFrame(
        {"delta_rmse_new_minus_parent": [0.251, -0.20]}
    )
    failed = evaluate_scientific_gate(
        new_metrics=new_metrics,
        new_fold=new_fold,
        parent_fold=parent_fold,
        pooled_bucket=pooled_bucket,
        hidden=hidden,
        by_well=failed_by_well,
        gate=config["gates"]["scientific"],
    )
    assert failed["passed"] is False
    assert failed["checks"]["worst_well_non_regression"] is False


def test_compact_notebook_is_readable_import_safe_and_not_canonical_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = NOTEBOOK_SOURCE.read_text()
    for heading in (
        "## Contents",
        "## 3. Candidate, compute, leakage, and weight contract",
        "## 5. Stage A raw-test-safe feature freeze",
        "## 6. Fold-safe inverse-RMSE weighted Stage B",
        "## 7. Parent v5 comparison and all-AND gate",
        "## 8. Metrics, diagnostics, feature importance, and generated artifacts",
    ):
        assert heading in source
    assert "candidate_task_weight_config=CONFIG[\"candidate_task_weight\"]" in source
    assert "run_stage_b(" in source
    assert "evaluate_exp407_stage_b(" in source
    assert "__file__" not in source
    assert (EXP_DIR / f"{EXP_NAME}_train.ipynb").is_file()

    monkeypatch.setenv("EXP407_IMPORT_ONLY", "1")
    namespace = runpy.run_path(str(NOTEBOOK_SOURCE))
    assert namespace["EXECUTE_NOTEBOOK"] is False
    assert namespace["static_contract"]["cost"]["planned_cpu_boosters"] == 10


def test_shared_stage_b_weight_hook_is_explicit_and_validation_unweighted() -> None:
    source = (
        ROOT / "src" / "candidate_selector_pipeline.py"
    ).read_text()
    stage_b = source[source.index("def run_stage_b(") : source.index("def run_stage_c(")]
    assert "candidate_task_weight_config: Mapping[str, Any] | None = None" in stage_b
    assert stage_b.count('{"sample_weight": train_sample_weight}') == 2
    assert "eval_sample_weight" not in stage_b
    assert '"validation_sample_weight_applied": False' in stage_b
