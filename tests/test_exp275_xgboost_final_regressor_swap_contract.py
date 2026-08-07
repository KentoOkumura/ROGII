from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "experiments" / "exp275_xgboost_final_regressor_swap_on_exp238"
CONFIG = yaml.safe_load((EXP_DIR / "config.yaml").read_text())
PUBLIC_NOTEBOOK = ROOT / CONFIG["model"]["public_source"]["local_path"]
TRAIN_SCRIPT = EXP_DIR / (
    "exp275_xgboost_final_regressor_swap_on_exp238_compact_selfcontained_train.py"
)
INFERENCE_SCRIPT = EXP_DIR / (
    "exp275_xgboost_final_regressor_swap_on_exp238_compact_selfcontained_inference.py"
)


def _eval_public_ast(node: ast.AST, env: dict[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return env[node.id]
    if isinstance(node, ast.IfExp):
        return _eval_public_ast(
            node.body if _eval_public_ast(node.test, env) else node.orelse,
            env,
        )
    if isinstance(node, ast.Dict):
        return {
            _eval_public_ast(key, env): _eval_public_ast(value, env)
            for key, value in zip(node.keys, node.values, strict=True)
        }
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval_public_ast(node.operand, env)
    raise AssertionError(ast.dump(node))


def _public_params() -> dict[str, Any]:
    notebook = json.loads(PUBLIC_NOTEBOOK.read_text())
    env: dict[str, Any] = {"FAST_DEBUG": False, "RANDOM_STATE": 42}
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell.get("source", []))
        if "XGB_PARAMS" not in source:
            continue
        for statement in ast.parse(source).body:
            if not isinstance(statement, ast.Assign):
                continue
            for target in statement.targets:
                if isinstance(target, ast.Name) and target.id in {
                    "FAST_DEBUG",
                    "RANDOM_STATE",
                }:
                    env[target.id] = _eval_public_ast(statement.value, env)
                if isinstance(target, ast.Name) and target.id == "XGB_PARAMS":
                    return dict(_eval_public_ast(statement.value, env))
    raise AssertionError("XGB_PARAMS was not found")


def test_public_notebook_sha_and_params_are_exact() -> None:
    assert (
        hashlib.sha256(PUBLIC_NOTEBOOK.read_bytes()).hexdigest()
        == CONFIG["model"]["public_source"]["sha256"]
    )
    assert _public_params() == CONFIG["model"]["xgboost"]["source_exact_params"]


def test_gpu_cost_and_frozen_control_contract() -> None:
    model = CONFIG["model"]
    assert model["active_variant_count"] == 1
    assert model["xgboost_config_count"] == 1
    assert model["folds"] == 5
    assert model["total_new_boosters"] == 5
    assert model["parent_control_retraining"] is False
    assert model["selector_retraining"] is False
    assert model["sample_weight"] == "none"
    assert model["run_approved"] is True
    assert model["approval"]["status"] == "approved"
    assert model["approval"]["approved_scope"] == (
        "one_variant_one_public_xgboost_config_five_folds_five_boosters_no_control_retraining"
    )
    assert (
        f'if nested(CONFIG, "model.approval.status") != "{model["approval"]["status"]}":'
    ) in TRAIN_SCRIPT.read_text()
    assert model["xgboost"]["source_exact_fit_params"]["early_stopping_rounds"] is None


def test_feature_and_parent_oof_contract() -> None:
    model = CONFIG["model"]
    parent = CONFIG["frozen_parent"]
    assert model["expected_base_feature_count"] == 380
    assert model["expected_selector_feature_count"] == 35
    assert model["expected_final_feature_count"] == 415
    assert parent["expected_lgb_mean_rmse"] == 7.936689853668213
    assert parent["expected_oof_decompressed_sha256"] == (
        "0e7390ac3b3a432b1d432e432cb374cbf38da393a9b95f8f0d6c22732030010c"
    )
    assert len(parent["selector_score_sha256_decompressed"]) == 5


def test_train_script_has_one_public_xgboost_fit_without_tuning() -> None:
    source = TRAIN_SCRIPT.read_text()
    assert source.count("model = XGBRegressor(**public_params)") == 1
    assert "early_stopping_rounds=" not in source
    assert "sample_weight=" not in source
    assert "xgboost_parameter_grid" not in source
    assert "parent_control_retraining" in source
    assert "selector_retraining" in source
    assert "model.total_new_boosters" in source


def test_canonical_notebooks_are_readable_and_reference_scoring_is_guarded() -> None:
    train = json.loads(
        (EXP_DIR / "exp275_xgboost_final_regressor_swap_on_exp238_train.ipynb").read_text()
    )
    inference = json.loads(
        (EXP_DIR / "exp275_xgboost_final_regressor_swap_on_exp238_inference.ipynb").read_text()
    )
    train_text = "\n".join("".join(cell.get("source", [])) for cell in train["cells"])
    inference_text = "\n".join("".join(cell.get("source", [])) for cell in inference["cells"])
    assert "## Contents" in train_text
    assert "## 6. Public-parameter XGBoost training" in train_text
    assert "XGBRegressor" in train_text
    assert "## 6. Saved XGBoost and parent-reference inference" in inference_text
    assert "user_authorized_reference_inference_and_scoring_2026_07_18" in (inference_text)
    assert "submission.csv" in inference_text
    assert "raw_xgboost_only" in inference_text
    assert ".fit(" not in INFERENCE_SCRIPT.read_text()
    assert CONFIG["inference"]["training_during_inference"] is False
    assert CONFIG["inference"]["selector_training_during_inference"] is False
    assert CONFIG["inference"]["expected_xgboost_model_count"] == 5
    assert CONFIG["inference"]["competition_submit_requested"] is True
    assert 'sort_values("feature_index")' in INFERENCE_SCRIPT.read_text()
