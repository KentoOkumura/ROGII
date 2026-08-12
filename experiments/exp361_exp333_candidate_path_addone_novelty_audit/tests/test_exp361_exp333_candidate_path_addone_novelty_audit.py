from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = ROOT / "experiments/exp361_exp333_candidate_path_addone_novelty_audit"
TRAIN_SOURCE = (
    EXP_DIR
    / "exp361_exp333_candidate_path_addone_novelty_audit_compact_selfcontained_train.py"
)
INFERENCE_SOURCE = (
    EXP_DIR
    / "exp361_exp333_candidate_path_addone_novelty_audit_compact_selfcontained_inference.py"
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def train_module():
    return load_module("exp361_train_test_module", TRAIN_SOURCE)


@pytest.fixture(scope="module")
def inference_module():
    return load_module("exp361_inference_test_module", INFERENCE_SOURCE)


@pytest.fixture(scope="module")
def config() -> dict:
    return yaml.safe_load((EXP_DIR / "config.yaml").read_text())


def test_fixed_execution_contract(train_module, config) -> None:
    train_module.validate_execution_contract(
        config, require_kaggle_authorization=True
    )
    assert config["experiment"]["route"] == "ensemble"
    assert train_module.EXPECTED_VARIANTS == {"exp333_segment_offset": 16}
    assert config["execution"]["total_boosters"] == 0
    assert config["execution"]["parent_or_control_regeneration"] is False


def test_exp333_pre_freeze_allowlist_excludes_truth(config) -> None:
    allowlist = set(config["data"]["exp333_oof"]["pre_freeze_columns"])
    forbidden = set(config["data"]["exp333_oof"]["forbidden_pre_freeze_columns"])
    assert allowlist == {
        "well_id",
        "row_idx",
        "outer_fold",
        "tvt_pred_stage1",
    }
    assert not allowlist & forbidden
    assert "tvt_true" not in allowlist
    assert (
        config["data"]["exp333_oof"]["upstream_reported_prediction_sha256"]
        == "dbb3f41642a2d6a9da704d276ed6398b706059078bcfcaca95e17e5c7af00784"
    )


def novelty_frame(
    *,
    h512_improvement: float,
    whole_improvement: float,
    unique_fraction: float,
    improved_folds: int,
) -> pd.DataFrame:
    records = [
        {
            "variant": "exp333_segment_offset",
            "granularity": "h512",
            "scope": "pooled",
            "fold": None,
            "base_oracle_rmse": 3.68,
            "add_one_oracle_rmse": 3.68 - h512_improvement,
            "oracle_improvement_ft": h512_improvement,
            "strict_unique_best_fraction": unique_fraction,
        },
        {
            "variant": "exp333_segment_offset",
            "granularity": "whole_well",
            "scope": "pooled",
            "fold": None,
            "base_oracle_rmse": 4.78,
            "add_one_oracle_rmse": 4.78 - whole_improvement,
            "oracle_improvement_ft": whole_improvement,
            "strict_unique_best_fraction": unique_fraction,
        },
    ]
    for fold in range(5):
        improvement = 0.01 if fold < improved_folds else 0.0
        records.append(
            {
                "variant": "exp333_segment_offset",
                "granularity": "h512",
                "scope": "fold",
                "fold": fold,
                "base_oracle_rmse": 3.68,
                "add_one_oracle_rmse": 3.68 - improvement,
                "oracle_improvement_ft": improvement,
                "strict_unique_best_fraction": unique_fraction,
            }
        )
    return pd.DataFrame(records)


def test_novelty_gate_passes_only_registered_contract(train_module, config) -> None:
    passed = train_module.evaluate_novelty_guards(
        novelty_frame(
            h512_improvement=0.03,
            whole_improvement=0.02,
            unique_fraction=0.02,
            improved_folds=4,
        ),
        config,
    )
    assert passed["passed"] is True

    failed = train_module.evaluate_novelty_guards(
        novelty_frame(
            h512_improvement=0.029,
            whole_improvement=0.02,
            unique_fraction=0.02,
            improved_folds=4,
        ),
        config,
    )
    assert failed["passed"] is False


def test_direct_scores_are_context_only(config) -> None:
    decision = config["success_criteria"]["decision"]
    assert decision["direct_scores_are_gate"] is False
    assert decision["pass"] == "exp333_candidate_path_novelty_supported"
    assert decision["fail"] == "close_exp333_candidate_novelty_branch"


def test_inference_remains_fail_closed(inference_module, config) -> None:
    checks = inference_module.validate_disabled_inference(config)
    assert all(checks.values())
    assert config["runtime"]["kaggle"]["inference_kernel_sources"] == []
    assert "raw_test_inference" in config["forbidden_actions"]
    assert "submission" in config["forbidden_actions"]
