from __future__ import annotations

import ast
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments" / "exp251_raw_test_safe_dual_objective_candidate_ranker"
CONFIG = EXP / "config.yaml"
SOURCE = EXP / "raw_test_safe_dual_objective_candidate_ranker.py"
ENGINE = EXP / "candidate_ranker_engine.py"


def _function(path: Path, name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function not found: {name}")


def test_active_stage_trains_corrected_variant_after_same_run_audit() -> None:
    config = yaml.safe_load(CONFIG.read_text())

    execution = config["execution"]
    assert execution["stage"] == "train_after_feature_audit"
    assert set(execution["allowed_stages"]) == {
        "feature_audit_only",
        "train_after_feature_audit",
    }
    assert execution["train_requires_same_run_audit_pass"] is True
    assert config["model"]["active_variants"] == ["raw_test_regenerated_copcf"]
    assert config["model"]["planned_lightgbm_configs"] == 2
    assert config["model"]["planned_folds"] == 5
    assert config["model"]["planned_boosters"] == 10
    assert config["model"]["control_retraining"] is False
    assert config["model"]["parent_retraining"] is False
    assert config["augmentation"]["enabled"] is False
    assert config["inference"]["submission"] is False


def test_copcf_features_are_regenerated_and_train_only_auxiliaries_are_disallowed() -> None:
    config = yaml.safe_load(CONFIG.read_text())
    audit = config["feature_audit"]

    assert audit["expected_parent_long_feature_count"] == 297
    assert audit["expected_selected_long_feature_count"] == 295
    assert audit["expected_regenerated_prefix_feature_count"] == 165
    assert audit["disallowed_prefixes"] == []
    assert audit["regenerated_prefixes"] == ["copcf_"]
    assert set(audit["disallowed_exact_columns"]) == {
        "exp226_gr_delta",
        "exp226_geop_tvt",
        "exp226_geop_minus_pred",
        "exp226_geop_minus_pred_abs",
    }
    assert "raw_test_candidate_derivation" in audit["allowed_provenance"]
    assert (
        "raw_test_full_train_prior_or_target_free_cluster"
        in audit["allowed_provenance"]
    )


def test_training_path_runs_same_call_feature_audit_first() -> None:
    function = _function(SOURCE, "run_raw_test_safe_candidate_ranker")
    calls = [
        node.func.id
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]

    assert calls.count("run_feature_audit") == 1
    assert calls.count("run_training_after_feature_audit") == 1

    training_source = ast.get_source_segment(
        SOURCE.read_text(), _function(SOURCE, "run_training_after_feature_audit")
    )
    assert training_source is not None
    assert "same-run feature audit failed" in training_source
    assert "len(model_manifest) != 10" in training_source


def test_engine_accepts_only_an_explicit_selected_schema() -> None:
    function_source = ast.get_source_segment(
        ENGINE.read_text(), _function(ENGINE, "train_outer_oof_models")
    )
    assert function_source is not None
    assert "selected_feature_columns" in function_source
    assert "selected raw-test-safe features are missing" in function_source
