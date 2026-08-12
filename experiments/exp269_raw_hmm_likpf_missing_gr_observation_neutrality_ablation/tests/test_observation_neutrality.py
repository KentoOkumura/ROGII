from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml


EXP_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = EXP_DIR.parents[1]


def load_hmm_module():
    path = EXP_DIR / "exact_hmm_smoother.py"
    spec = importlib.util.spec_from_file_location("exp269_exact_hmm_smoother", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_stage1_guard_function():
    path = EXP_DIR / "exp269_raw_hmm_likpf_missing_gr_observation_neutrality_ablation_train.py"
    tree = ast.parse(path.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "evaluate_stage1_guard"
    )
    namespace = {"Any": Any, "np": np, "pd": pd}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["evaluate_stage1_guard"]


def normalized_function_ast(path: Path, name: str) -> str:
    tree = ast.parse(path.read_text())
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name
    )
    function.decorator_list = []
    return ast.dump(function, include_attributes=False)


def test_missing_gr_rows_are_state_neutral_and_observed_rows_are_unchanged() -> None:
    module = load_hmm_module()
    gr = np.array([10.0, 20.0, 30.0], dtype=np.float64)
    grid = np.array([5.0, 15.0, 25.0, 35.0], dtype=np.float64)
    missing = np.array([False, True, False])

    control = module.build_gr_emission_loglik(
        gr,
        grid,
        10.0,
        emission="gauss",
        df=4.0,
        raw_gr_missing=missing,
        mask_missing_gr=False,
    )
    neutral = module.build_gr_emission_loglik(
        gr,
        grid,
        10.0,
        emission="gauss",
        df=4.0,
        raw_gr_missing=missing,
        mask_missing_gr=True,
    )

    np.testing.assert_array_equal(neutral[~missing], control[~missing])
    np.testing.assert_array_equal(neutral[missing], np.zeros((1, len(grid)), np.float32))
    assert not np.array_equal(neutral[missing], control[missing])


def test_missing_gr_neutrality_requires_an_exact_row_mask() -> None:
    module = load_hmm_module()
    with pytest.raises(ValueError, match="requires raw_gr_missing"):
        module.build_gr_emission_loglik(
            np.array([10.0]),
            np.array([5.0, 15.0]),
            10.0,
            emission="gauss",
            df=4.0,
            mask_missing_gr=True,
        )
    with pytest.raises(ValueError, match="length mismatch"):
        module.build_gr_emission_loglik(
            np.array([10.0]),
            np.array([5.0, 15.0]),
            10.0,
            emission="gauss",
            df=4.0,
            raw_gr_missing=np.array([True, False]),
            mask_missing_gr=True,
        )


def test_scientific_wrapper_rejects_auxiliary_unary_overrides() -> None:
    module = load_hmm_module()
    with pytest.raises(ValueError, match="forbids override keys"):
        module.run_raw_hmm_missing_gr_neutral(
            None,
            None,
            lgb_tvt=np.array([1.0]),
        )


def test_exp209_hmm_settings_and_pf_fail_closed_contract() -> None:
    config = yaml.safe_load((EXP_DIR / "config.yaml").read_text())
    parent_config = yaml.safe_load(
        (
            REPO_ROOT
            / "experiments"
            / "exp209_exp072_exp205_joint_exact_parity_fast_cache_generation"
            / "config.yaml"
        ).read_text()
    )
    parent_metrics = yaml.safe_load(
        (
            REPO_ROOT
            / "experiments"
            / "exp209_exp072_exp205_joint_exact_parity_fast_cache_generation"
            / "metrics.json"
        ).read_text()
    )

    assert config["lineage"]["parent"] == (
        "exp209_exp072_exp205_joint_exact_parity_fast_cache_generation"
    )
    assert config["experiment"]["route"] == "pf_beam"
    assert config["model"]["hmm"] == parent_config["model"]["hmm"]
    assert "lgb_emission" not in config
    assert config["model"]["gr_missing"]["self_gr_unary"] is False
    assert config["model"]["gr_missing"]["lightgbm_unary"] is False
    assert config["pf_stage"]["enabled"] is False
    assert config["pf_stage"]["requires_separate_user_approval"] is True
    assert config["inference"]["enabled"] is False
    assert config["data"]["control_expected_decompressed_sha256"] == parent_metrics[
        "hmm_generated_decompressed_sha256"
    ]


def test_exp209_transition_and_prefix_statistics_code_are_unchanged() -> None:
    child = EXP_DIR / "exact_hmm_smoother.py"
    parent = (
        REPO_ROOT
        / "experiments"
        / "exp209_exp072_exp205_joint_exact_parity_fast_cache_generation"
        / "exact_hmm_smoother.py"
    )
    assert normalized_function_ast(child, "prefix_stats") == normalized_function_ast(
        parent, "prefix_stats"
    )
    assert normalized_function_ast(child, "_hmm2_fb") == normalized_function_ast(
        parent, "_hmm2_fb"
    )


def test_stage1_guard_passes_complete_evidence_and_fails_closed_on_missing_group() -> None:
    evaluate = load_stage1_guard_function()
    group_metrics = pd.DataFrame(
        [
            ("overall", "all", -0.03),
            ("gr_availability", "missing", -0.01),
            ("gr_availability", "observed", 0.01),
            ("distance", "1000_plus", 0.01),
            ("hidden_like", "spatial", 0.01),
            ("hidden_like", "purged", 0.01),
        ],
        columns=["group_type", "group", "delta_rmse_neutral_minus_control"],
    )
    by_well = pd.DataFrame({"delta_rmse_neutral_minus_control": [0.24, -0.1]})
    finite = pd.DataFrame(
        {
            "candidate": ["raw_hmm_missing_gr_neutral"],
            "rows": [100],
            "prediction_finite_rows": [100],
            "std_finite_rows": [100],
        }
    )
    thresholds = {
        "overall_delta_rmse_max_ft": -0.02,
        "raw_missing_delta_rmse_max_ft": 0.0,
        "observed_delta_rmse_max_ft": 0.02,
        "distance_1000_plus_delta_rmse_max_ft": 0.02,
        "hidden_like_delta_rmse_max_ft": 0.02,
        "worst_well_delta_rmse_max_ft": 0.25,
        "prediction_finite_coverage_min": 1.0,
        "std_finite_coverage_min": 1.0,
        "id_mismatch_max": 0,
    }

    passed = evaluate(
        group_metrics,
        by_well,
        finite,
        id_mismatch_count=0,
        thresholds=thresholds,
        hidden_like_groups=["spatial", "purged"],
    )
    assert passed["passed"] is True
    assert passed["pf_stage_eligible"] is True
    assert passed["pf_stage_executed"] is False

    missing_group = evaluate(
        group_metrics[group_metrics["group"] != "purged"],
        by_well,
        finite,
        id_mismatch_count=0,
        thresholds=thresholds,
        hidden_like_groups=["spatial", "purged"],
    )
    assert missing_group["passed"] is False
    assert missing_group["pf_stage_eligible"] is False
