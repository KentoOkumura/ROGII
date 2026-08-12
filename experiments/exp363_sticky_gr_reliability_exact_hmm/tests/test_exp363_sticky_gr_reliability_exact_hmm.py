from __future__ import annotations

import copy
import importlib.util
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = ROOT / "experiments" / "exp363_sticky_gr_reliability_exact_hmm"
TRAIN_SOURCE = (
    EXP_DIR
    / "exp363_sticky_gr_reliability_exact_hmm_compact_selfcontained_train.py"
)
INFERENCE_SOURCE = (
    EXP_DIR
    / "exp363_sticky_gr_reliability_exact_hmm_compact_selfcontained_inference.py"
)
CONFIG_PATH = EXP_DIR / "config.yaml"


def load_module(path: Path, name: str):
    old_value = os.environ.get("EXP363_IMPORT_ONLY")
    os.environ["EXP363_IMPORT_ONLY"] = "1"
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if old_value is None:
            os.environ.pop("EXP363_IMPORT_ONLY", None)
        else:
            os.environ["EXP363_IMPORT_ONLY"] = old_value


def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())


def test_frozen_config_and_completed_zero_cost_stage0_contract() -> None:
    train = load_module(TRAIN_SOURCE, "exp363_train_contract")
    config = load_config()
    train.validate_scientific_contract(config)
    assert config["experiment"]["status"] == "stage_0_failed_closed"
    assert config["execution"]["run_stage_0"] is False
    assert config["execution"]["run_stage_1"] is False
    assert config["execution"]["kaggle_push_approved"] is False
    assert config["execution"]["kaggle_kernel_status"] == "COMPLETE"
    assert config["execution_contract"]["stage_0"] == {
        "diagnostic_variants": 1,
        "reporting_folds": 5,
        "hmm_well_runs": 0,
        "model_configs": 0,
        "lightgbm_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
    }
    assert config["execution_contract"]["parent_control_retraining"] is False
    assert config["results"]["technical_gate_passed"] is True
    assert config["results"]["scientific_gate_passed"] is False
    assert config["results"]["stage_1_eligible"] is False


def test_sticky_forward_filter_respects_fixed_initial_and_persistence() -> None:
    train = load_module(TRAIN_SOURCE, "exp363_train_forward")
    transition = np.asarray(
        [[511.0 / 512.0, 1.0 / 512.0], [1.0 / 128.0, 127.0 / 128.0]]
    )
    initial = np.asarray([0.8, 0.2])
    multipliers = np.asarray([1.0, 0.25])

    neutral = train.sticky_forward_filter(
        np.zeros(8), transition, initial, multipliers
    )
    np.testing.assert_allclose(neutral, 0.2, atol=1.0e-15, rtol=0.0)

    surprised = train.sticky_forward_filter(
        np.asarray([-20.0, 0.0, 0.0]), transition, initial, multipliers
    )
    assert surprised[0] > 0.99
    assert surprised[1] > 0.98
    assert surprised[2] > 0.97


def test_block_policy_keeps_short_tail_and_stride_overlap() -> None:
    train = load_module(TRAIN_SOURCE, "exp363_train_blocks")
    config = load_config()
    rows = pd.DataFrame(
        {
            "id": [f"well0001_{index}" for index in range(700)],
            "well_id": "well0001",
            "row_idx": np.arange(700, dtype=np.int64),
            "suffix_offset": np.arange(700, dtype=np.int64),
            "raw_gr_observed": 1,
            "exp209_sigma": 20.0,
            "path_log_emission": np.linspace(0.0, -3.0, 700),
        }
    )
    ledger, posterior = train.build_well_block_features(rows, config)
    assert ledger["start_suffix_offset"].tolist() == [0, 256, 512]
    assert ledger["block_row_count"].tolist() == [512, 444, 188]
    assert posterior["block_id"].tolist() == [0, 1, 2]
    assert posterior["weak_posterior_mean"].between(0.0, 1.0).all()


def test_circular_offset_is_stable_and_nonzero_for_multi_block_well() -> None:
    train = load_module(TRAIN_SOURCE, "exp363_train_circular")
    first = train.stable_circular_offset("well0001", 17, "exp363")
    second = train.stable_circular_offset("well0001", 17, "exp363")
    assert first == second
    assert 1 <= first <= 16
    assert train.stable_circular_offset("well0001", 1, "exp363") == 0


def test_auc_handles_ties_and_one_class_scope() -> None:
    train = load_module(TRAIN_SOURCE, "exp363_train_auc")
    assert train.roc_auc_binary(
        np.asarray([False, False, True, True]),
        np.asarray([0.0, 0.0, 1.0, 1.0]),
    ) == pytest.approx(1.0)
    assert (
        train.roc_auc_binary(
            np.asarray([True, True]), np.asarray([0.0, 1.0])
        )
        is None
    )


def passing_gate_inputs(train):
    records = []
    for fold in range(5):
        for local_well in range(2):
            well_id = f"well_{fold}_{local_well}"
            for block_id, (score, circular, rmse, bad, quartile) in enumerate(
                [(0.1, 0.9, 5.0, False, 1), (0.9, 0.1, 15.0, True, 4)]
            ):
                records.append(
                    {
                        "well_id": well_id,
                        "block_id": block_id,
                        "fold": fold,
                        "weak_posterior_mean": score,
                        "weak_posterior_sum": score * 512.0,
                        "circular_weak_score": circular,
                        "circular_offset_blocks": 1,
                        "block_row_count": 512,
                        "block_rmse": rmse,
                        "bad10": bad,
                        "weak_quartile": quartile,
                        "hidden_like_spatial": True,
                        "hidden_like_typewell_purged": True,
                    }
                )
    block_readout = pd.DataFrame(records)
    config = copy.deepcopy(load_config())
    config["validation"]["expected_wells"] = 10
    config["validation"]["expected_rows"] = 20
    scopes = train.build_scope_metrics(block_readout, config)
    frozen = {
        "freeze": {
            "strict_quartile_boundaries": True,
            "truth_columns_read_before_freeze": 0,
        }
    }
    preflight = {"saved_path_frame": pd.DataFrame(index=np.arange(20))}
    return block_readout, scopes, frozen, preflight, config


def test_stage0_gate_requires_all_fixed_readouts() -> None:
    train = load_module(TRAIN_SOURCE, "exp363_train_gate")
    block_readout, scopes, frozen, preflight, config = passing_gate_inputs(train)
    gate = train.evaluate_stage_0_gate(
        block_readout, scopes, frozen, preflight, config
    )
    assert gate["technical_pass"] is True
    assert gate["scientific_pass"] is True
    assert gate["passed"] is True
    assert gate["scientific"]["passing_folds_auc_gt_0p50"] == 5

    broken_scopes = scopes.copy()
    broken_scopes.loc[
        broken_scopes["scope"] == "hidden_like_spatial", "real_bad10_auc"
    ] = 0.54
    gate = train.evaluate_stage_0_gate(
        block_readout, broken_scopes, frozen, preflight, config
    )
    assert gate["scientific_pass"] is False
    assert gate["stage_1_eligible"] is False


def test_inference_is_fail_closed_and_never_copies_sample_submission() -> None:
    inference = load_module(INFERENCE_SOURCE, "exp363_inference_contract")
    config = load_config()
    contract = inference.validate_disabled_inference(config)
    assert contract["hmm_well_runs"] == 0
    with pytest.raises(RuntimeError, match="Stage 1 exact HMM"):
        inference.stop_disabled_inference(config)
    source = INFERENCE_SOURCE.read_text()
    assert "shutil.copyfile" not in source
    assert "submission.csv" not in source


def test_source_is_not_a_thin_helper_entrypoint() -> None:
    source = TRAIN_SOURCE.read_text()
    headings = [
        "## 2. Runtime, configuration, path, and SHA helpers",
        "## 3. Frozen scientific and execution contract",
        "## 4. Input preflight and target-free exp209 path preparation",
        "## 5. Sticky reliability forward filter and block freeze",
        "## 6. Late truth, fold, and hidden-like attachment",
        "## 7. AUC, quartile, fold, and promotion-gate readout",
        "## 8. Metrics, diagnostics, and generated artifacts",
        "## 9. Setup and configuration preview",
        "## 10. Run the approved Kaggle CPU Stage 0",
    ]
    for heading in headings:
        assert heading in source
    assert "from settings import" not in source
    assert "__file__" not in source
